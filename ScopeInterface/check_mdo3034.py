#!/usr/bin/env python3
"""Acquire analog waveforms from a Tektronix MDO3034 through PyVISA."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    import pyvisa
    from pyvisa.errors import Error as VisaError
except ImportError as exc:
    raise SystemExit(
        "PyVISA is required. Install it and a VISA backend before running this script."
    ) from exc


EXPECTED_VENDOR = "TEKTRONIX"
EXPECTED_MODEL = "MDO3034"
VISA_RESOURCE = "USB0::1689::1032::C053047::0::INSTR"
WAVEFORM_SOURCE = "CH1"
WAVEFORMS_TO_READ = 3_000
CAPTURE_RATE_HZ = 10000.0
# Horizontal time scale in seconds per division.
HORIZONTAL_SCALE_S_PER_DIV = 100e-9
SAVE_TIME_MIN_S = -500e-9
SAVE_TIME_MAX_S = 500e-9
WAVEFORM_CSV_DIR = Path("mdo3034_ch1_waveforms_1p2_kV")
CAPTURE_TIMES_FILE = Path("mdo3034_ch1_capture_times_1p2_kV.csv")
METADATA_FILE = Path("mdo3034_ch1_metadata.json")
TIMEOUT_MS = 10_000


def configure_scope(scope: Any) -> None:
    scope.timeout = TIMEOUT_MS
    scope.read_termination = "\n"
    scope.write_termination = "\n"
    scope.write(f"HORIZONTAL:SCALE {HORIZONTAL_SCALE_S_PER_DIV}")


def is_mdo3034(idn: str) -> bool:
    fields = [field.strip().upper() for field in idn.split(",")]
    return (
        len(fields) >= 2
        and fields[0] == EXPECTED_VENDOR
        and fields[1] == EXPECTED_MODEL
    )


def configure_waveform_transfer(scope: Any) -> None:
    scope.write(f"DATA:SOURCE {WAVEFORM_SOURCE}")
    scope.write("DATA:ENC RIBINARY")
    scope.write("DATA:WIDTH 1")


def read_scale(scope: Any) -> dict[str, float]:
    return {
        "horizontal_scale_s_per_div": float(scope.query("HORIZONTAL:SCALE?")),
        "xincr": float(scope.query("WFMOUTPRE:XINCR?")),
        "xzero": float(scope.query("WFMOUTPRE:XZERO?")),
        "ymult": float(scope.query("WFMOUTPRE:YMULT?")),
        "yzero": float(scope.query("WFMOUTPRE:YZERO?")),
    }


def wait_for_single_acquisition(scope: Any) -> None:
    scope.write("ACQUIRE:STOPAFTER SEQUENCE")
    scope.write("ACQUIRE:STATE RUN")
    scope.query("*OPC?")


def read_raw_waveform(scope: Any) -> list[int]:
    return scope.query_binary_values("CURVE?", datatype="b", container=list)


def waveform_csv_path(capture_index: int) -> Path:
    return WAVEFORM_CSV_DIR / f"waveform_{capture_index:05d}.csv"


def write_waveform_csv(
    capture_index: int, raw_waveform: list[int], scale: dict[str, float]
) -> int:
    points_saved = 0
    with waveform_csv_path(capture_index).open("w", newline="") as waveform_file:
        writer = csv.writer(waveform_file)
        writer.writerow(("sample_index", "time_s", "voltage_v", "raw_digitizing_level"))

        for sample_index, raw_sample in enumerate(raw_waveform):
            seconds = scale["xzero"] + sample_index * scale["xincr"]
            if seconds < SAVE_TIME_MIN_S or seconds > SAVE_TIME_MAX_S:
                continue

            volts = scale["yzero"] + raw_sample * scale["ymult"]
            writer.writerow((sample_index, seconds, volts, raw_sample))
            points_saved += 1

    return points_saved


def pace_next_capture(started_at: float, capture_index: int) -> tuple[float, float]:
    period_s = 1.0 / CAPTURE_RATE_HZ
    scheduled_at = started_at + capture_index * period_s
    delay_s = scheduled_at - time.monotonic()
    if delay_s > 0:
        time.sleep(delay_s)
    return scheduled_at, max(0.0, -delay_s)


def acquire_run(scope: Any) -> tuple[dict[str, float], float]:
    configure_waveform_transfer(scope)
    scale = read_scale(scope)
    WAVEFORM_CSV_DIR.mkdir(exist_ok=True)
    max_late_s = 0.0
    started_at = time.monotonic()

    with CAPTURE_TIMES_FILE.open("w", newline="") as capture_times_file:
        writer = csv.writer(capture_times_file)
        writer.writerow(
            (
                "waveform_index",
                "scheduled_s",
                "completed_s",
                "late_s",
                "points_returned",
                "points_saved",
            )
        )

        for capture_index in range(WAVEFORMS_TO_READ):
            scheduled_at, late_s = pace_next_capture(started_at, capture_index)
            wait_for_single_acquisition(scope)
            raw_waveform = read_raw_waveform(scope)
            points_saved = write_waveform_csv(capture_index, raw_waveform, scale)
            completed_at = time.monotonic()
            writer.writerow(
                (
                    capture_index,
                    scheduled_at,
                    completed_at,
                    late_s,
                    len(raw_waveform),
                    points_saved,
                )
            )
            max_late_s = max(max_late_s, late_s)

            if (capture_index + 1) % 100 == 0:
                print(f"Captured {capture_index + 1}/{WAVEFORMS_TO_READ} waveforms.")

    return scale, max_late_s


def write_metadata(idn: str, scale: dict[str, float], max_late_s: float) -> None:
    metadata = {
        "idn": idn,
        "visa_resource": VISA_RESOURCE,
        "waveform_source": WAVEFORM_SOURCE,
        "waveforms": WAVEFORMS_TO_READ,
        "points_per_waveform": "scope_returned",
        "requested_capture_rate_hz": CAPTURE_RATE_HZ,
        "horizontal_scale_s_per_div": HORIZONTAL_SCALE_S_PER_DIV,
        "saved_time_window_s": [SAVE_TIME_MIN_S, SAVE_TIME_MAX_S],
        "waveform_csv_dir": str(WAVEFORM_CSV_DIR),
        "waveform_csv_name_pattern": "waveform_00000.csv",
        "capture_times_file": str(CAPTURE_TIMES_FILE),
        "waveform_csv_columns": [
            "sample_index",
            "time_s",
            "voltage_v",
            "raw_digitizing_level",
        ],
        "max_start_late_s": max_late_s,
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
                scale, max_late_s = acquire_run(scope)
            finally:
                scope.write("ACQUIRE:STOPAFTER RUNSTOP")
    except (RuntimeError, ValueError, VisaError) as exc:
        print(f"Could not acquire from {VISA_RESOURCE}: {exc}", file=sys.stderr)
        return 1
    finally:
        manager.close()

    write_metadata(idn, scale, max_late_s)
    print("OK: Tektronix MDO3034 run completed.")
    print(
        f"Saved {WAVEFORMS_TO_READ} {WAVEFORM_SOURCE} waveform CSV files "
        f"to {WAVEFORM_CSV_DIR}."
    )
    print(f"Capture timing: {CAPTURE_TIMES_FILE}. Metadata: {METADATA_FILE}.")
    if max_late_s > 0:
        print(
            f"Maximum requested {CAPTURE_RATE_HZ:g} Hz start overrun: "
            f"{max_late_s:.6f} s."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
