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

SAMPLE_LABEL = "HYBRID_PMT_1p3kv_calibration_100min_DETEN_no_PPO"
EXPECTED_VENDOR = "TEKTRONIX"
EXPECTED_MODEL = "MDO3034"
VISA_RESOURCE = "USB0::1689::1032::C053047::0::INSTR"
BASE_DIR = Path(__file__).resolve().parent
ACQUISITION_DURATION_MINUTES = 100.0
ACQUISITION_MODE = "HIRES"
TRANSFER_WIDTH_BYTES = 2
CSV_FILE = BASE_DIR / f"waveforms_{SAMPLE_LABEL.lower()}.csv"
ROOT_FILE = BASE_DIR / f"waveforms_{SAMPLE_LABEL.lower()}.root"
METADATA_FILE = BASE_DIR / f"metadata_{SAMPLE_LABEL.lower()}.json"
TIMEOUT_MS = 30_000
DATE_COLLECTED = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
WAVEFORM_COLUMNS = (
    "sample_index",
    "time_s",
    "voltage_v",
    "raw_digitizing_level",
)
PROGRESS_BAR_WIDTH = 30


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
    scope.write(f"DATA:WIDTH {TRANSFER_WIDTH_BYTES}")


def configure_waveform_record_range(scope: Any) -> None:
    scope.write("DATA:START 1")
    try:
        record_length = int(float(scope.query("HORIZONTAL:RECORDLENGTH?")))
    except (ValueError, VisaError):
        return
    scope.write(f"DATA:STOP {record_length}")


def configure_acquisition(scope: Any) -> None:
    scope.write(f"ACQUIRE:MODE {ACQUISITION_MODE}")


def query_float(scope: Any, command: str) -> float | None:
    try:
        return float(scope.query(command))
    except (ValueError, VisaError):
        return None


def query_int(scope: Any, command: str) -> int | None:
    value = query_float(scope, command)
    if value is None:
        return None
    return int(value)


def query_str(scope: Any, command: str) -> str | None:
    try:
        return scope.query(command).strip()
    except VisaError:
        return None


def read_trigger_settings(scope: Any) -> dict[str, Any]:
    trigger: dict[str, Any] = {}
    trigger["level_v"] = query_float(scope, "TRIGGER:A:LEVEL?")
    trigger["source"] = query_str(scope, "TRIGGER:A:EDGE:SOURCE?")
    trigger["type"] = query_str(scope, "TRIGGER:A:TYPE?")
    trigger["slope"] = query_str(scope, "TRIGGER:A:EDGE:SLOPE?")
    return trigger


def read_channel_settings(scope: Any, source: str) -> dict[str, Any]:
    channel = {
        "source": source,
        "scale_v_per_div": query_float(scope, f"{source}:SCALE?"),
        "position_div": query_float(scope, f"{source}:POSITION?"),
        "offset_v": query_float(scope, f"{source}:OFFSET?"),
        "coupling": query_str(scope, f"{source}:COUPLING?"),
        "termination_ohm": query_float(scope, f"{source}:TERMINATION?"),
        "bandwidth": query_str(scope, f"{source}:BANDWIDTH?"),
        "invert": query_int(scope, f"{source}:INVERT?"),
        "probe_gain": query_float(scope, f"{source}:PROBE?"),
    }
    return {key: value for key, value in channel.items() if value is not None}


def read_transfer_settings(scope: Any) -> dict[str, Any]:
    transfer = {
        "data_source": query_str(scope, "DATA:SOURCE?"),
        "data_start": query_int(scope, "DATA:START?"),
        "data_stop": query_int(scope, "DATA:STOP?"),
        "data_encoding": query_str(scope, "DATA:ENCdg?"),
        "data_width_bytes": query_int(scope, "DATA:WIDTH?"),
        "wfmoutpre": query_str(scope, "WFMOUTPRE?"),
    }
    return {key: value for key, value in transfer.items() if value is not None}


def read_acquisition_settings(scope: Any) -> dict[str, Any]:
    acquisition = {
        "mode": query_str(scope, "ACQUIRE:MODE?"),
        "num_average": query_int(scope, "ACQUIRE:NUMAVG?"),
        "sample_rate_s_per_s": query_float(scope, "ACQUIRE:SRATE?"),
        "stop_after": query_str(scope, "ACQUIRE:STOPAFTER?"),
        "state": query_int(scope, "ACQUIRE:STATE?"),
    }
    return {key: value for key, value in acquisition.items() if value is not None}


def read_scale(scope: Any) -> dict[str, Any]:
    scale = {
        "acquisition_mode": scope.query("ACQUIRE:MODE?").strip(),
        "horizontal_scale_s_per_div": float(scope.query("HORIZONTAL:SCALE?")),
        "xincr": float(scope.query("WFMOUTPRE:XINCR?")),
        "xzero": float(scope.query("WFMOUTPRE:XZERO?")),
        "ymult": float(scope.query("WFMOUTPRE:YMULT?")),
        "yzero": float(scope.query("WFMOUTPRE:YZERO?")),
        "yoff": float(scope.query("WFMOUTPRE:YOFF?")),
        "transfer_width_bytes": float(scope.query("WFMOUTPRE:BYT_NR?")),
    }
    try:
        scale["byte_order"] = scope.query("WFMOUTPRE:BYT_OR?").strip().upper()
    except VisaError:
        scale["byte_order"] = "MSB"
    try:
        scale["record_length"] = float(scope.query("HORIZONTAL:RECORDLENGTH?"))
    except (ValueError, VisaError):
        pass
    try:
        scale["waveform_points"] = float(scope.query("WFMOUTPRE:NR_PT?"))
    except (ValueError, VisaError):
        pass
    try:
        scale["sample_rate_s_per_s"] = float(scope.query("ACQUIRE:SRATE?"))
    except (ValueError, VisaError):
        pass
    try:
        scale["horizontal_delay_time_s"] = float(scope.query("HORIZONTAL:DELAY:TIME?"))
        scale["horizontal_delay_mode"] = int(
            float(scope.query("HORIZONTAL:DELAY:MODE?"))
        )
    except (ValueError, VisaError):
        scale["horizontal_delay_time_s"] = 0.0
        scale["horizontal_delay_mode"] = 0
    return scale


def read_raw_waveform(scope: Any, scale: dict[str, Any]) -> list[int]:
    transfer_width = int(scale["transfer_width_bytes"])
    if transfer_width == 1:
        datatype = "b"
    elif transfer_width == 2:
        datatype = "h"
    else:
        raise ValueError(f"Unsupported waveform transfer width: {transfer_width}")

    return scope.query_binary_values(
        "CURVE?",
        datatype=datatype,
        is_big_endian=not str(scale.get("byte_order", "MSB")).startswith("LSB"),
        container=list,
    )


def wait_for_triggered_acquisition(scope: Any) -> None:
    scope.write("ACQUIRE:STOPAFTER SEQUENCE")
    scope.write("ACQUIRE:STATE RUN")
    scope.query("*OPC?")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def print_progress(completed: int, duration_s: float, started_at: float) -> None:
    elapsed_s = time.monotonic() - started_at
    fraction = min(elapsed_s / duration_s, 1.0) if duration_s > 0 else 1.0
    filled = round(PROGRESS_BAR_WIDTH * fraction)
    bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    remaining_s = max(0.0, duration_s - elapsed_s)

    print(
        f"\r[{bar}] {fraction * 100:6.2f}% "
        f"({completed} waveforms) "
        f"elapsed {format_duration(elapsed_s)} "
        f"left {format_duration(remaining_s)}",
        end="",
        flush=True,
    )


def prompt_yes_no(question: str) -> bool:
    while True:
        answer = input(f"{question} [y/n]: ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def waveform_arrays(
    raw_waveform: list[int], scale: dict[str, Any]
) -> dict[str, np.ndarray]:
    if not raw_waveform:
        raise ValueError("The scope returned an empty waveform.")

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

    if raw.size == 0:
        raise ValueError(
            "No samples were found inside the displayed time window. "
            "Check DATA:START/DATA:STOP, horizontal scale, and delay settings."
        )

    return {
        "sample_index": sample_index,
        "time_s": time_s.astype(np.float64),
        "voltage_v": (
            scale["yzero"] + (raw - scale["yoff"]) * scale["ymult"]
        ).astype(np.float64),
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


def empty_resolution_diagnostics() -> dict[str, Any]:
    return {
        "waveforms_checked": 0,
        "total_points_checked": 0,
        "raw_min": None,
        "raw_max": None,
        "min_raw_code_step": None,
        "raw_code_step_gcd": None,
        "min_observed_voltage_step_v": None,
        "raw_low_byte_values": set(),
    }


def update_resolution_diagnostics(
    diagnostics: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    raw = arrays["raw_digitizing_level"].astype(np.int64)
    voltage = arrays["voltage_v"].astype(np.float64)
    unique_raw = np.unique(raw)
    unique_voltage = np.unique(voltage)

    diagnostics["waveforms_checked"] += 1
    diagnostics["total_points_checked"] += int(raw.size)
    diagnostics["raw_min"] = (
        int(np.min(raw))
        if diagnostics["raw_min"] is None
        else min(diagnostics["raw_min"], int(np.min(raw)))
    )
    diagnostics["raw_max"] = (
        int(np.max(raw))
        if diagnostics["raw_max"] is None
        else max(diagnostics["raw_max"], int(np.max(raw)))
    )
    diagnostics["raw_low_byte_values"].update(int(value) for value in raw & 0xFF)

    raw_steps = np.diff(unique_raw)
    raw_steps = raw_steps[raw_steps > 0]
    if raw_steps.size:
        min_raw_step = int(np.min(raw_steps))
        diagnostics["min_raw_code_step"] = (
            min_raw_step
            if diagnostics["min_raw_code_step"] is None
            else min(diagnostics["min_raw_code_step"], min_raw_step)
        )
        step_gcd = int(np.gcd.reduce(raw_steps))
        diagnostics["raw_code_step_gcd"] = (
            step_gcd
            if diagnostics["raw_code_step_gcd"] is None
            else int(np.gcd(diagnostics["raw_code_step_gcd"], step_gcd))
        )

    voltage_steps = np.diff(unique_voltage)
    voltage_steps = voltage_steps[voltage_steps > 0]
    if voltage_steps.size:
        min_voltage_step = float(np.min(voltage_steps))
        diagnostics["min_observed_voltage_step_v"] = (
            min_voltage_step
            if diagnostics["min_observed_voltage_step_v"] is None
            else min(diagnostics["min_observed_voltage_step_v"], min_voltage_step)
        )


def finalize_resolution_diagnostics(
    diagnostics: dict[str, Any],
    scale: dict[str, Any],
) -> dict[str, Any]:
    finalized = dict(diagnostics)
    low_byte_values = sorted(finalized.pop("raw_low_byte_values"))
    finalized["raw_low_byte_values"] = low_byte_values
    finalized["raw_low_byte_always_zero"] = low_byte_values == [0]
    finalized["ymult_v_per_raw_count"] = scale["ymult"]
    if finalized["min_raw_code_step"] is not None:
        finalized["effective_step_from_raw_step_v"] = (
            finalized["min_raw_code_step"] * scale["ymult"]
        )
    if finalized["raw_code_step_gcd"] is not None:
        finalized["effective_step_from_raw_gcd_v"] = (
            finalized["raw_code_step_gcd"] * scale["ymult"]
        )
    return finalized


def acquire_displayed_waveforms(
    scope: Any,
) -> tuple[
    str,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    int,
    int,
    dict[str, Any],
]:
    if ACQUISITION_DURATION_MINUTES <= 0:
        raise ValueError("ACQUISITION_DURATION_MINUTES must be greater than zero.")

    source = visible_waveform_source(scope)
    configure_acquisition(scope)
    configure_waveform_transfer(scope, source)
    configure_waveform_record_range(scope)
    scale = read_scale(scope)
    trigger = read_trigger_settings(scope)
    run_settings = {
        "channel": read_channel_settings(scope, source),
        "transfer": read_transfer_settings(scope),
        "acquisition": read_acquisition_settings(scope),
    }

    points_saved = 0
    root_branches = {}
    first_waveform_arrays: dict[str, np.ndarray] | None = None
    resolution_diagnostics = empty_resolution_diagnostics()
    progress_started_at = time.monotonic()
    acquisition_started_at = time.time()
    acquisition_duration_s = ACQUISITION_DURATION_MINUTES * 60.0
    acquisition_deadline = progress_started_at + acquisition_duration_s
    interrupted = False

    try:
        while time.monotonic() < acquisition_deadline:
            waveform_index = len(root_branches)
            wait_for_triggered_acquisition(scope)
            raw_waveform = read_raw_waveform(scope, scale)
            arrays = waveform_arrays(raw_waveform, scale)
            update_resolution_diagnostics(resolution_diagnostics, arrays)

            if waveform_index == 0:
                points_saved = len(arrays["voltage_v"])
                first_waveform_arrays = arrays
            elif len(arrays["voltage_v"]) != points_saved:
                raise ValueError(
                    "ROOT output expects each saved waveform to have the same length."
                )

            root_branches[f"waveform_{waveform_index:05d}"] = arrays["voltage_v"]
            print_progress(
                waveform_index + 1,
                acquisition_duration_s,
                progress_started_at,
            )
    except KeyboardInterrupt:
        interrupted = True

    print()
    actual_duration_s = time.time() - acquisition_started_at
    waveforms_saved = len(root_branches)

    acquisition_timing = {
        "requested_duration_minutes": ACQUISITION_DURATION_MINUTES,
        "requested_duration_s": acquisition_duration_s,
        "actual_duration_s": actual_duration_s,
        "actual_duration_minutes": actual_duration_s / 60.0,
        "started_at": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(acquisition_started_at)
        ),
        "ended_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "completed_requested_duration": not interrupted,
        "interrupted": interrupted,
    }

    if interrupted:
        print(
            "Acquisition interrupted after "
            f"{format_duration(actual_duration_s)} with {waveforms_saved} waveforms."
        )
        if not root_branches:
            raise RuntimeError("No waveforms were collected before the interruption.")
        if not prompt_yes_no("Save the waveforms collected so far?"):
            raise RuntimeError("Acquisition interrupted; collected data was not saved.")
    elif not root_branches:
        raise RuntimeError("No waveforms were collected during the requested time.")

    if first_waveform_arrays is None:
        raise RuntimeError("No first waveform preview is available to save.")
    write_waveform_csv(first_waveform_arrays)

    with uproot.recreate(ROOT_FILE) as root_file:
        root_tree = root_file.mktree(
            "waveforms",
            {branch_name: np.float64 for branch_name in root_branches},
        )
        root_tree.extend(root_branches)

    run_settings["effective_resolution"] = finalize_resolution_diagnostics(
        resolution_diagnostics,
        scale,
    )
    channel_scale_v_per_div = run_settings["channel"].get("scale_v_per_div")
    if channel_scale_v_per_div is not None:
        run_settings["effective_resolution"]["estimated_8bit_screen_step_v"] = (
            10.0 * channel_scale_v_per_div / 256.0
        )
    return (
        source,
        scale,
        trigger,
        run_settings,
        points_saved,
        waveforms_saved,
        acquisition_timing,
    )


def write_metadata(
    idn: str,
    source: str,
    scale: dict[str, Any],
    trigger: dict[str, Any],
    run_settings: dict[str, Any],
    points_saved: int,
    waveforms_saved: int,
    acquisition_timing: dict[str, Any],
) -> None:
    metadata = {
        "date": DATE_COLLECTED,
        "idn": idn,
        "visa_resource": VISA_RESOURCE,
        "waveform_source": source,
        "waveforms": waveforms_saved,
        "points_per_waveform": points_saved,
        "csv_file": str(CSV_FILE),
        "root_file": str(ROOT_FILE),
        "root_tree": "waveforms",
        "root_entries": points_saved,
        "root_columns": [
            f"waveform_{index:05d}" for index in range(waveforms_saved)
        ],
        "root_units": "V",
        "waveform_csv_columns": list(WAVEFORM_COLUMNS),
        "acquisition_time": acquisition_timing,
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
        "trigger": trigger,
        "scope_settings": run_settings,
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
                (
                    source,
                    scale,
                    trigger,
                    run_settings,
                    points_saved,
                    waveforms_saved,
                    acquisition_timing,
                ) = (
                    acquire_displayed_waveforms(scope)
                )
            finally:
                scope.write("ACQUIRE:STOPAFTER RUNSTOP")
    except (RuntimeError, ValueError, VisaError) as exc:
        print(f"Could not acquire from {VISA_RESOURCE}: {exc}", file=sys.stderr)
        return 1
    finally:
        manager.close()

    write_metadata(
        idn,
        source,
        scale,
        trigger,
        run_settings,
        points_saved,
        waveforms_saved,
        acquisition_timing,
    )
    print("OK: Tektronix MDO3034 waveforms saved.")
    print(
        f"Source: {source}. Saved {waveforms_saved} waveforms "
        f"with {points_saved} points each in "
        f"{format_duration(acquisition_timing['actual_duration_s'])}."
    )
    print(f"Saved first waveform CSV preview to {CSV_FILE}.")
    print(f"Saved ROOT tree with all waveforms to {ROOT_FILE}.")
    print(f"Metadata: {METADATA_FILE}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
