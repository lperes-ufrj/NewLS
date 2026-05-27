#!/usr/bin/env python3
"""Save the waveform currently displayed on a Tektronix MDO3034."""

from __future__ import annotations
import time
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import uproot

try:
    import pyvisa
    from pyvisa.errors import Error as VisaError
except ImportError as exc:
    raise SystemExit(
        "PyVISA is required. Install it and a VISA backend before running this script."
    ) from exc

SAMPLE_LABEL = "HYBRID_PMT_CALIBRATION_1p3KV"
EXPECTED_VENDOR = "TEKTRONIX"
EXPECTED_MODEL = "MDO3034"
VISA_RESOURCE = "USB0::0x0699::0x0408::C053047::INSTR"
BASE_DIR = Path(__file__).resolve().parent
WAVEFORMS_TO_READ = 3_000
CSV_FILE = BASE_DIR / f"waveforms_{SAMPLE_LABEL.lower()}.csv"
ROOT_FILE = BASE_DIR / f"waveforms_{SAMPLE_LABEL.lower()}.root"
METADATA_FILE = BASE_DIR / f"metadata_{SAMPLE_LABEL.lower()}.json"
TIMEOUT_MS = 10_000
DATE_COLLECTED = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
WAVEFORM_COLUMNS = (
    "sample_index",
    "time_s",
    "voltage_v",
    "raw_digitizing_level",
)


def configure_scope(scope: Any) -> None:
    scope.timeout = TIMEOUT_MS
    scope.read_termination = "\n"
    scope.write_termination = "\n"


def is_mdo3034(idn: str) -> bool:
    fields = [field.strip().upper() for field in idn.split(",")]
    return (
        len(fields) >= 2
        and fields[0] == EXPECTED_VENDOR
        and fields[1] == EXPECTED_MODEL
    )


def visible_waveform_source(scope: Any) -> str:
    visible_channels = [
        channel
        for channel in ("CH1", "CH2", "CH3", "CH4")
        if int(float(scope.query(f"SELECT:{channel}?")))
    ]
    if len(visible_channels) == 1:
        return visible_channels[0]
    return scope.query("DATA:SOURCE?").strip()


def configure_waveform_transfer(scope: Any, source: str) -> None:
    scope.write(f"DATA:SOURCE {source}")
    scope.write("DATA:ENC RIBINARY")
    scope.write("DATA:WIDTH 1")


def read_scale(scope: Any) -> dict[str, float]:
    scale = {
        "horizontal_scale_s_per_div": float(scope.query("HORIZONTAL:SCALE?")),
        "xincr": float(scope.query("WFMOUTPRE:XINCR?")),
        "xzero": float(scope.query("WFMOUTPRE:XZERO?")),
        "ymult": float(scope.query("WFMOUTPRE:YMULT?")),
        "yzero": float(scope.query("WFMOUTPRE:YZERO?")),
    }
    try:
        scale["horizontal_delay_time_s"] = float(scope.query("HORIZONTAL:DELAY:TIME?"))
        scale["horizontal_delay_mode"] = int(
            float(scope.query("HORIZONTAL:DELAY:MODE?"))
        )
    except (ValueError, VisaError):
        scale["horizontal_delay_time_s"] = 0.0
        scale["horizontal_delay_mode"] = 0
    return scale


def read_raw_waveform(scope: Any) -> list[int]:
    return scope.query_binary_values("CURVE?", datatype="b", container=list)


def wait_for_triggered_acquisition(scope: Any) -> None:
    scope.write("ACQUIRE:STOPAFTER SEQUENCE")
    scope.write("ACQUIRE:STATE RUN")
    scope.query("*OPC?")


def waveform_arrays(
    raw_waveform: list[int], scale: dict[str, float]
) -> dict[str, np.ndarray]:
    raw = np.asarray(raw_waveform, dtype=np.int16)
    sample_index = np.arange(raw.size, dtype=np.int32)
    time_s = scale["xzero"] + sample_index * scale["xincr"]
    display_width_s = 10.0 * scale["horizontal_scale_s_per_div"]
    display_center_s = (
        scale["horizontal_delay_time_s"] if scale["horizontal_delay_mode"] else 0.0
    )
    display_min_s = display_center_s - display_width_s / 2.0
    display_max_s = display_center_s + display_width_s / 2.0
    visible = (display_min_s <= time_s) & (time_s <= display_max_s)

    raw = raw[visible]
    sample_index = sample_index[visible]
    time_s = time_s[visible]

    return {
        "sample_index": sample_index,
        "time_s": time_s.astype(np.float64),
        "voltage_v": (scale["yzero"] + raw * scale["ymult"]).astype(np.float64),
        "raw_digitizing_level": raw,
    }


def write_waveform_csv(arrays: dict[str, np.ndarray]) -> int:
    with CSV_FILE.open("w", newline="") as waveform_file:
        writer = csv.writer(waveform_file)
        writer.writerow(WAVEFORM_COLUMNS)
        writer.writerows(
            zip(
                arrays["sample_index"],
                arrays["time_s"],
                arrays["voltage_v"],
                arrays["raw_digitizing_level"],
            )
        )
    return len(arrays["sample_index"])


def acquire_displayed_waveforms(scope: Any) -> tuple[str, dict[str, float], int]:
    source = visible_waveform_source(scope)
    configure_waveform_transfer(scope, source)
    scale = read_scale(scope)

    points_saved = 0
    root_branches = {}

    for waveform_index in range(WAVEFORMS_TO_READ):
        wait_for_triggered_acquisition(scope)
        raw_waveform = read_raw_waveform(scope)
        arrays = waveform_arrays(raw_waveform, scale)

        if waveform_index == 0:
            points_saved = len(arrays["voltage_v"])
            write_waveform_csv(arrays)
        elif len(arrays["voltage_v"]) != points_saved:
            raise ValueError(
                "ROOT output expects each saved waveform to have the same length."
            )

        root_branches[f"waveform_{waveform_index:05d}"] = arrays["voltage_v"]

        if (waveform_index + 1) % 100 == 0:
            print(f"Captured {waveform_index + 1}/{WAVEFORMS_TO_READ} waveforms.")

    with uproot.recreate(ROOT_FILE) as root_file:
        root_tree = root_file.mktree(
            "waveforms",
            {branch_name: np.float64 for branch_name in root_branches},
        )
        root_tree.extend(root_branches)

    return source, scale, points_saved


def write_metadata(
    idn: str, source: str, scale: dict[str, float], points_saved: int
) -> None:
    metadata = {
        "date": DATE_COLLECTED,
        "idn": idn,
        "visa_resource": VISA_RESOURCE,
        "waveform_source": source,
        "waveforms": WAVEFORMS_TO_READ,
        "points_per_waveform": points_saved,
        "csv_file": str(CSV_FILE),
        "root_file": str(ROOT_FILE),
        "root_tree": "waveforms",
        "root_entries": points_saved,
        "root_columns": [f"waveform_{index:05d}" for index in range(WAVEFORMS_TO_READ)],
        "root_units": "V",
        "waveform_csv_columns": list(WAVEFORM_COLUMNS),
        "display_time_window_s": [
            (
                scale["horizontal_delay_time_s"]
                if scale["horizontal_delay_mode"]
                else 0.0
            )
            - 5.0 * scale["horizontal_scale_s_per_div"],
            (
                scale["horizontal_delay_time_s"]
                if scale["horizontal_delay_mode"]
                else 0.0
            )
            + 5.0 * scale["horizontal_scale_s_per_div"],
        ],
        "scale": scale,
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> int:
    try:
        manager = pyvisa.ResourceManager()
    except (OSError, ValueError, VisaError) as exc:
        print(f"Could not initialize a VISA resource manager: {exc}", file=sys.stderr)
        return 2

    try:
        with manager.open_resource(VISA_RESOURCE) as scope:
            configure_scope(scope)
            idn = scope.query("*IDN?").strip()
            print(f"{VISA_RESOURCE}: {idn}")
            if not is_mdo3034(idn):
                print(
                    "The responding instrument is not a Tektronix MDO3034.",
                    file=sys.stderr,
                )
                return 1

            try:
                source, scale, points_saved = acquire_displayed_waveforms(scope)
            finally:
                scope.write("ACQUIRE:STOPAFTER RUNSTOP")
    except (RuntimeError, ValueError, VisaError) as exc:
        print(f"Could not acquire from {VISA_RESOURCE}: {exc}", file=sys.stderr)
        return 1
    finally:
        manager.close()

    write_metadata(idn, source, scale, points_saved)
    print("OK: Tektronix MDO3034 waveforms saved.")
    print(
        f"Source: {source}. Saved {WAVEFORMS_TO_READ} waveforms "
        f"with {points_saved} points each."
    )
    print(f"Saved first waveform CSV preview to {CSV_FILE}.")
    print(f"Saved ROOT tree with all waveforms to {ROOT_FILE}.")
    print(f"Metadata: {METADATA_FILE}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
