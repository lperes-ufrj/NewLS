#!/usr/bin/env python3
"""Acquire triggered waveforms from a Keysight InfiniiVision DSOX1204A.

The first waveform is written to CSV, every waveform is written to a ROOT
tree, and the oscilloscope setup and waveform preamble are written to JSON.
Edit the constants below for a run, or pass --resource/--source/--minutes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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


SAMPLE_LABEL = "HYBRID_PMT_1p2kv_cs137_keysight_50min_1_1_NORMAL_detalp_no_ppo_run2"
# Replace this with the value printed by ResourceManager().list_resources(), or
# supply it at run time with --resource.
VISA_RESOURCE = "USB0::10893::902::CN61306140::0::INSTR"
EXPECTED_VENDOR = "KEYSIGHT TECHNOLOGIES"
EXPECTED_MODEL = "DSOX1204A"
ACQUISITION_DURATION_MINUTES = 50.0
ACQUISITION_TYPE = "NORMAL"#"HRESOLUTION"
TRANSFER_FORMAT = "WORD"
TIMEOUT_MS = 30_000
PROGRESS_BAR_WIDTH = 30
BASE_DIR = Path(__file__).resolve().parent
WAVEFORM_COLUMNS = (
    "sample_index",
    "time_s",
    "voltage_v",
    "raw_digitizing_level",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource", default=VISA_RESOURCE, help="VISA resource string")
    parser.add_argument(
        "--source",
        choices=("CHAN1", "CHAN2", "CHAN3", "CHAN4"),
        help="analog channel (default: the only displayed channel, or the first one)",
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=ACQUISITION_DURATION_MINUTES,
        help="acquisition duration in minutes",
    )
    parser.add_argument("--label", default=SAMPLE_LABEL, help="output file label")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR,
        help="directory for CSV, ROOT, and JSON files",
    )
    return parser.parse_args()


def output_paths(output_dir: Path, label: str) -> tuple[Path, Path, Path]:
    stem = label.lower()
    return (
        output_dir / f"waveforms_{stem}.csv",
        output_dir / f"waveforms_{stem}.root",
        output_dir / f"metadata_{stem}.json",
    )


def configure_scope(scope: Any) -> None:
    scope.timeout = TIMEOUT_MS
    scope.read_termination = "\n"
    scope.write_termination = "\n"


def is_dsox1204a(idn: str) -> bool:
    fields = [field.strip().upper() for field in idn.split(",")]
    return (
        len(fields) >= 2
        and fields[0] in {EXPECTED_VENDOR, "AGILENT TECHNOLOGIES"}
        and fields[1] == EXPECTED_MODEL
    )


def query_float(scope: Any, command: str) -> float | None:
    try:
        return float(scope.query(command))
    except (ValueError, VisaError):
        return None


def query_int(scope: Any, command: str) -> int | None:
    value = query_float(scope, command)
    return None if value is None else int(value)


def query_str(scope: Any, command: str) -> str | None:
    try:
        return scope.query(command).strip()
    except VisaError:
        return None


def visible_waveform_source(scope: Any, requested_source: str | None) -> str:
    if requested_source is not None:
        return requested_source
    visible = [
        channel
        for channel in ("CHAN1", "CHAN2", "CHAN3", "CHAN4")
        if query_int(scope, f":{channel}:DISPLAY?") == 1
    ]
    if not visible:
        raise RuntimeError("No displayed analog channel was found.")
    if len(visible) > 1:
        print(f"Multiple channels are displayed; using {visible[0]}.")
    return visible[0]


def configure_acquisition(scope: Any) -> None:
    scope.write(f":ACQUIRE:TYPE {ACQUISITION_TYPE}")


def configure_waveform_transfer(scope: Any, source: str) -> None:
    scope.write(f":WAVEFORM:SOURCE {source}")
    scope.write(f":WAVEFORM:FORMAT {TRANSFER_FORMAT}")
    scope.write(":WAVEFORM:BYTEORDER LSBFIRST")
    scope.write(":WAVEFORM:UNSIGNED ON")
    # RAW returns acquisition-memory data; MAX requests all points available.
    scope.write(":WAVEFORM:POINTS:MODE RAW")
    scope.write(":WAVEFORM:POINTS MAX")


def parse_preamble(preamble: str) -> dict[str, Any]:
    fields = [field.strip() for field in preamble.split(",")]
    if len(fields) < 10:
        raise ValueError(f"Unexpected waveform preamble: {preamble!r}")
    values = [float(field) for field in fields[:10]]
    return {
        "format_code": int(values[0]),
        "acquisition_type_code": int(values[1]),
        "points": int(values[2]),
        "average_count": int(values[3]),
        "xincr": values[4],
        "xorigin": values[5],
        "xreference": values[6],
        "yincr": values[7],
        "yorigin": values[8],
        "yreference": values[9],
        "raw": preamble,
    }


def read_scale(scope: Any) -> dict[str, Any]:
    scale = parse_preamble(scope.query(":WAVEFORM:PREAMBLE?").strip())
    scale["acquisition_type"] = query_str(scope, ":ACQUIRE:TYPE?")
    scale["horizontal_scale_s_per_div"] = query_float(scope, ":TIMEBASE:SCALE?")
    scale["horizontal_position_s"] = query_float(scope, ":TIMEBASE:POSITION?")
    scale["sample_rate_s_per_s"] = query_float(scope, ":ACQUIRE:SRATE?")
    scale["waveform_points"] = query_int(scope, ":WAVEFORM:POINTS?")
    scale["transfer_format"] = query_str(scope, ":WAVEFORM:FORMAT?")
    scale["byte_order"] = query_str(scope, ":WAVEFORM:BYTEORDER?")
    return {key: value for key, value in scale.items() if value is not None}


def read_trigger_settings(scope: Any) -> dict[str, Any]:
    trigger = {
        "mode": query_str(scope, ":TRIGGER:MODE?"),
        "source": query_str(scope, ":TRIGGER:EDGE:SOURCE?"),
        "level_v": query_float(scope, ":TRIGGER:EDGE:LEVEL?"),
        "slope": query_str(scope, ":TRIGGER:EDGE:SLOPE?"),
        "sweep": query_str(scope, ":TRIGGER:SWEEP?"),
    }
    return {key: value for key, value in trigger.items() if value is not None}


def read_channel_settings(scope: Any, source: str) -> dict[str, Any]:
    channel = {
        "source": source,
        "displayed": query_int(scope, f":{source}:DISPLAY?"),
        "scale_v_per_div": query_float(scope, f":{source}:SCALE?"),
        "offset_v": query_float(scope, f":{source}:OFFSET?"),
        "coupling": query_str(scope, f":{source}:COUPLING?"),
        "impedance_ohm": query_float(scope, f":{source}:IMPEDANCE?"),
        "bandwidth_limit": query_str(scope, f":{source}:BWLIMIT?"),
        "invert": query_int(scope, f":{source}:INVERT?"),
        "probe_ratio": query_float(scope, f":{source}:PROBE?"),
    }
    return {key: value for key, value in channel.items() if value is not None}


def read_transfer_settings(scope: Any) -> dict[str, Any]:
    transfer = {
        "source": query_str(scope, ":WAVEFORM:SOURCE?"),
        "format": query_str(scope, ":WAVEFORM:FORMAT?"),
        "byte_order": query_str(scope, ":WAVEFORM:BYTEORDER?"),
        "unsigned": query_int(scope, ":WAVEFORM:UNSIGNED?"),
        "points_mode": query_str(scope, ":WAVEFORM:POINTS:MODE?"),
        "points": query_int(scope, ":WAVEFORM:POINTS?"),
    }
    return {key: value for key, value in transfer.items() if value is not None}


def read_acquisition_settings(scope: Any) -> dict[str, Any]:
    acquisition = {
        "type": query_str(scope, ":ACQUIRE:TYPE?"),
        "count": query_int(scope, ":ACQUIRE:COUNT?"),
        "sample_rate_s_per_s": query_float(scope, ":ACQUIRE:SRATE?"),
    }
    return {key: value for key, value in acquisition.items() if value is not None}


def wait_for_triggered_acquisition(scope: Any, source: str) -> None:
    # DIGitize clears acquisition memory, waits for a trigger, acquires one
    # waveform, and stops. *OPC? prevents the following transfer from racing it.
    scope.write(f":DIGITIZE {source}")
    scope.query("*OPC?")


def read_raw_waveform(scope: Any) -> list[int]:
    return scope.query_binary_values(
        ":WAVEFORM:DATA?",
        datatype="H",
        is_big_endian=False,
        container=list,
    )


def waveform_arrays(
    raw_waveform: list[int], scale: dict[str, Any]
) -> dict[str, np.ndarray]:
    if not raw_waveform:
        raise ValueError("The scope returned an empty waveform.")

    raw = np.asarray(raw_waveform, dtype=np.uint16)
    sample_index = np.arange(raw.size, dtype=np.int32)
    time_s = (
        (sample_index.astype(np.float64) - scale["xreference"]) * scale["xincr"]
        + scale["xorigin"]
    )
    voltage_v = (
        (raw.astype(np.float64) - scale["yreference"]) * scale["yincr"]
        + scale["yorigin"]
    )
    return {
        "sample_index": sample_index,
        "time_s": time_s,
        "voltage_v": voltage_v,
        "raw_digitizing_level": raw,
    }


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
    fraction = min(elapsed_s / duration_s, 1.0) if duration_s else 1.0
    filled = round(PROGRESS_BAR_WIDTH * fraction)
    bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    print(
        f"\r[{bar}] {fraction * 100:6.2f}% ({completed} waveforms) "
        f"elapsed {format_duration(elapsed_s)} "
        f"left {format_duration(max(0.0, duration_s - elapsed_s))}",
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


def write_waveform_csv(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="") as waveform_file:
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


def acquire_displayed_waveforms(
    scope: Any,
    source: str,
    duration_minutes: float,
    csv_file: Path,
    root_file: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int, int, dict[str, Any]]:
    if duration_minutes <= 0:
        raise ValueError("--minutes must be greater than zero.")

    configure_acquisition(scope)
    configure_waveform_transfer(scope, source)
    trigger = read_trigger_settings(scope)
    run_settings = {
        "channel": read_channel_settings(scope, source),
        "transfer": read_transfer_settings(scope),
        "acquisition": read_acquisition_settings(scope),
    }

    root_branches: dict[str, np.ndarray] = {}
    first_arrays: dict[str, np.ndarray] | None = None
    scale: dict[str, Any] | None = None
    points_saved = 0
    started_monotonic = time.monotonic()
    started_epoch = time.time()
    duration_s = duration_minutes * 60.0
    interrupted = False

    try:
        while time.monotonic() - started_monotonic < duration_s:
            index = len(root_branches)
            wait_for_triggered_acquisition(scope, source)
            # DIGitize can change the number of returned points, so configure
            # transfer and read the preamble after each acquisition.
            configure_waveform_transfer(scope, source)
            current_scale = read_scale(scope)
            arrays = waveform_arrays(read_raw_waveform(scope), current_scale)
            if first_arrays is None:
                first_arrays = arrays
                scale = current_scale
                points_saved = len(arrays["voltage_v"])
            elif len(arrays["voltage_v"]) != points_saved:
                raise ValueError(
                    "ROOT output expects each saved waveform to have the same length."
                )
            root_branches[f"waveform_{index:05d}"] = arrays["voltage_v"]
            print_progress(index + 1, duration_s, started_monotonic)
    except KeyboardInterrupt:
        interrupted = True

    print()
    actual_duration_s = time.time() - started_epoch
    timing = {
        "requested_duration_minutes": duration_minutes,
        "requested_duration_s": duration_s,
        "actual_duration_s": actual_duration_s,
        "actual_duration_minutes": actual_duration_s / 60.0,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started_epoch)),
        "ended_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "completed_requested_duration": not interrupted,
        "interrupted": interrupted,
    }
    if interrupted:
        print(
            f"Acquisition interrupted after {format_duration(actual_duration_s)} "
            f"with {len(root_branches)} waveforms."
        )
        if not root_branches:
            raise RuntimeError("No waveforms were collected before the interruption.")
        if not prompt_yes_no("Save the waveforms collected so far?"):
            raise RuntimeError("Acquisition interrupted; collected data was not saved.")
    elif not root_branches:
        raise RuntimeError("No waveforms were collected during the requested time.")

    if first_arrays is None or scale is None:
        raise RuntimeError("No first waveform is available to save.")
    write_waveform_csv(csv_file, first_arrays)
    with uproot.recreate(root_file) as root:
        tree = root.mktree(
            "waveforms",
            {name: np.float64 for name in root_branches},
        )
        tree.extend(root_branches)

    # The previous script reports useful voltage-resolution context. Keysight's
    # preamble directly supplies the actual volts-per-code value.
    run_settings["effective_resolution"] = {
        "yincr_v_per_raw_count": scale["yincr"],
        "transfer_format": scale.get("transfer_format", TRANSFER_FORMAT),
    }
    return (
        scale,
        trigger,
        run_settings,
        points_saved,
        len(root_branches),
        timing,
    )


def write_metadata(
    path: Path,
    *,
    idn: str,
    resource: str,
    source: str,
    csv_file: Path,
    root_file: Path,
    scale: dict[str, Any],
    trigger: dict[str, Any],
    run_settings: dict[str, Any],
    points_saved: int,
    waveforms_saved: int,
    timing: dict[str, Any],
) -> None:
    metadata = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "idn": idn,
        "visa_resource": resource,
        "waveform_source": source,
        "waveforms": waveforms_saved,
        "points_per_waveform": points_saved,
        "csv_file": str(csv_file),
        "root_file": str(root_file),
        "root_tree": "waveforms",
        "root_entries": points_saved,
        "root_columns": [
            f"waveform_{index:05d}" for index in range(waveforms_saved)
        ],
        "root_units": "V",
        "waveform_csv_columns": list(WAVEFORM_COLUMNS),
        "acquisition_time": timing,
        "scale": scale,
        "trigger": trigger,
        "scope_settings": run_settings,
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    csv_file, root_file, metadata_file = output_paths(args.output_dir, args.label)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        manager = pyvisa.ResourceManager()
    except (OSError, ValueError, VisaError) as exc:
        print(f"Could not initialize a VISA resource manager: {exc}", file=sys.stderr)
        return 2

    try:
        with manager.open_resource(args.resource) as scope:
            configure_scope(scope)
            idn = scope.query("*IDN?").strip()
            print(f"{args.resource}: {idn}")
            if not is_dsox1204a(idn):
                print(
                    "The responding instrument is not a Keysight DSOX1204A.",
                    file=sys.stderr,
                )
                return 1
            source = visible_waveform_source(scope, args.source)
            (
                scale,
                trigger,
                run_settings,
                points_saved,
                waveforms_saved,
                timing,
            ) = acquire_displayed_waveforms(
                scope,
                source,
                args.minutes,
                csv_file,
                root_file,
            )
    except (RuntimeError, ValueError, VisaError) as exc:
        print(f"Could not acquire from {args.resource}: {exc}", file=sys.stderr)
        return 1
    finally:
        manager.close()

    write_metadata(
        metadata_file,
        idn=idn,
        resource=args.resource,
        source=source,
        csv_file=csv_file,
        root_file=root_file,
        scale=scale,
        trigger=trigger,
        run_settings=run_settings,
        points_saved=points_saved,
        waveforms_saved=waveforms_saved,
        timing=timing,
    )
    print("OK: Keysight DSOX1204A waveforms saved.")
    print(
        f"Source: {source}. Saved {waveforms_saved} waveforms with "
        f"{points_saved} points each in {format_duration(timing['actual_duration_s'])}."
    )
    print(f"First waveform CSV: {csv_file}")
    print(f"ROOT tree: {root_file}")
    print(f"Metadata: {metadata_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
