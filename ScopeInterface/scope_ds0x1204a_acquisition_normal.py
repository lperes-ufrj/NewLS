#!/usr/bin/env python3
"""Acquire PMT waveforms with segmented self-triggering on a DSOX1204A.

The recommended acquisition uses NORMAL acquisition type, NORMAL edge
triggering, the horizontal window already configured on the oscilloscope,
and logical batches of 500 events. The DSOX1204A supports at most 50 hardware
memory segments, so each logical batch is acquired as ten consecutive
50-segment banks.

The first waveform is written to CSV, every waveform is written to a ROOT
tree, and the oscilloscope setup and waveform preamble are written to JSON.
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


SAMPLE_LABEL = "HYBRID_PMT_DSOX1204A"
# Replace this with the value printed by ResourceManager().list_resources(), or
# supply it at run time with --resource.
VISA_RESOURCE = "USB0::10893::902::CN61306140::0::INSTR"
EXPECTED_VENDOR = "KEYSIGHT TECHNOLOGIES"
EXPECTED_MODEL = "DSOX1204A"
ACQUISITION_DURATION_MINUTES = 100.0
ACQUISITION_TYPE = "NORMAL"
TARGET_EVENTS = 50_000
LOGICAL_SEGMENTED_BATCH_EVENTS = 500
HARDWARE_SEGMENTS_PER_BANK = 50
TRIGGER_FRACTION_OF_SPE = 0.25
PULSE_POLARITY = "NEGATIVE"
TRANSFER_FORMAT = "WORD"
TIMEOUT_MS = 120_000
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
        help="maximum acquisition duration in minutes",
    )
    parser.add_argument(
        "--events",
        type=int,
        default=TARGET_EVENTS,
        help="target event count (recommended: 50000 to 100000)",
    )
    parser.add_argument(
        "--waveform-ns",
        type=float,
        help=(
            "optionally override the scope's total waveform length in ns; "
            "omit this and --pretrigger-ns to use the scope window"
        ),
    )
    parser.add_argument(
        "--pretrigger-ns",
        type=float,
        help=(
            "optionally override the scope's pre-trigger interval in ns; "
            "omit this and --waveform-ns to use the scope window"
        ),
    )
    parser.add_argument(
        "--spe-amplitude-v",
        type=float,
        help=(
            "optionally derive and set the trigger level from a positive "
            "SPE pulse-amplitude magnitude in volts"
        ),
    )
    parser.add_argument(
        "--trigger-level-v",
        type=float,
        help=(
            "optionally override the scope's absolute trigger level in volts; "
            "omit this and --spe-amplitude-v to use the level set on the scope"
        ),
    )
    parser.add_argument(
        "--trigger-fraction",
        type=float,
        default=TRIGGER_FRACTION_OF_SPE,
        help="hardware threshold as a fraction of SPE amplitude (0.2 to 0.3)",
    )
    parser.add_argument(
        "--baseline-v",
        type=float,
        default=0.0,
        help="channel baseline voltage used to derive the trigger level",
    )
    parser.add_argument(
        "--pulse-polarity",
        choices=("NEGATIVE", "POSITIVE"),
        default=PULSE_POLARITY,
        help="PMT pulse polarity at the oscilloscope input",
    )
    parser.add_argument(
        "--high-voltage-kv",
        type=float,
        help="PMT high voltage in kV, recorded in metadata",
    )
    parser.add_argument("--label", default=SAMPLE_LABEL, help="output file label")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR,
        help="directory for CSV, ROOT, and JSON files",
    )
    args = parser.parse_args()

    if args.minutes <= 0:
        parser.error("--minutes must be greater than zero")
    if args.events <= 0:
        parser.error("--events must be greater than zero")
    if (args.waveform_ns is None) != (args.pretrigger_ns is None):
        parser.error(
            "provide both --waveform-ns and --pretrigger-ns, or omit both "
            "to use the window configured on the oscilloscope"
        )
    if args.waveform_ns is not None:
        if not 200.0 <= args.waveform_ns <= 500.0:
            parser.error("--waveform-ns must be between 200 and 500 ns")
        if not 50.0 <= args.pretrigger_ns <= 100.0:
            parser.error("--pretrigger-ns must be between 50 and 100 ns")
        if args.pretrigger_ns >= args.waveform_ns:
            parser.error("--pretrigger-ns must be shorter than --waveform-ns")
    if not 0.2 <= args.trigger_fraction <= 0.3:
        parser.error("--trigger-fraction must be between 0.2 and 0.3")
    if args.spe_amplitude_v is not None and args.spe_amplitude_v <= 0:
        parser.error("--spe-amplitude-v must be positive")
    if args.spe_amplitude_v is not None and args.trigger_level_v is not None:
        parser.error(
            "use either --spe-amplitude-v or --trigger-level-v, not both"
        )
    if args.high_voltage_kv is not None and args.high_voltage_kv <= 0:
        parser.error("--high-voltage-kv must be positive")
    if not 50_000 <= args.events <= 100_000:
        print(
            "Warning: recommended statistics are 50000 to 100000 events.",
            file=sys.stderr,
        )

    return args


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


def trigger_level_from_args(args: argparse.Namespace) -> float | None:
    """Return a requested hardware trigger level, or None to preserve it."""

    if args.trigger_level_v is not None:
        return float(args.trigger_level_v)
    if args.spe_amplitude_v is None:
        return None

    pulse_sign = -1.0 if args.pulse_polarity == "NEGATIVE" else 1.0
    return (
        float(args.baseline_v)
        + pulse_sign
        * float(args.trigger_fraction)
        * float(args.spe_amplitude_v)
    )


def configure_acquisition(
    scope: Any,
    source: str,
    *,
    waveform_length_ns: float | None,
    pretrigger_ns: float | None,
    trigger_level_v: float | None,
    pulse_polarity: str,
) -> None:
    """Configure recommended segmented NORMAL self-trigger acquisition."""

    scope.write(":STOP")
    scope.write(f":ACQUIRE:TYPE {ACQUISITION_TYPE}")
    scope.write(":ACQUIRE:MODE SEGMENTED")

    if waveform_length_ns is not None and pretrigger_ns is not None:
        waveform_length_s = waveform_length_ns * 1.0e-9
        pretrigger_s = pretrigger_ns * 1.0e-9
        # The display has ten horizontal divisions. With LEFT reference,
        # POSITION is the interval from the left reference to the trigger.
        scope.write(":TIMEBASE:MODE MAIN")
        scope.write(":TIMEBASE:REFERENCE LEFT")
        scope.write(f":TIMEBASE:SCALE {waveform_length_s / 10.0:.12g}")
        scope.write(f":TIMEBASE:POSITION {pretrigger_s:.12g}")

    scope.write(":TRIGGER:MODE EDGE")
    scope.write(f":TRIGGER:EDGE:SOURCE {source}")
    scope.write(f":TRIGGER:EDGE:SLOPE {pulse_polarity}")
    if trigger_level_v is not None:
        scope.write(f":TRIGGER:EDGE:LEVEL {trigger_level_v:.12g}")
    scope.write(":TRIGGER:SWEEP NORMAL")

    actual_mode = query_str(scope, ":ACQUIRE:MODE?")
    if actual_mode is None or not actual_mode.upper().startswith("SEGM"):
        error = query_str(scope, ":SYSTEM:ERROR?")
        raise RuntimeError(
            "The oscilloscope did not enter segmented mode. "
            "Verify that the SGM segmented-memory license is installed. "
            f"Instrument response: {actual_mode!r}; error: {error!r}"
        )


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
    scale["horizontal_reference"] = query_str(scope, ":TIMEBASE:REFERENCE?")
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
        "mode": query_str(scope, ":ACQUIRE:MODE?"),
        "type": query_str(scope, ":ACQUIRE:TYPE?"),
        "count": query_int(scope, ":ACQUIRE:COUNT?"),
        "segmented_count_setting": query_int(
            scope, ":ACQUIRE:SEGMENTED:COUNT?"
        ),
        "segments_acquired": query_int(
            scope, ":WAVEFORM:SEGMENTED:COUNT?"
        ),
        "sample_rate_s_per_s": query_float(scope, ":ACQUIRE:SRATE?"),
    }
    return {key: value for key, value in acquisition.items() if value is not None}


def acquire_segment_bank(scope: Any, source: str, count: int) -> int:
    """Arm one segmented bank and return the number of captured segments."""

    if not 2 <= count <= HARDWARE_SEGMENTS_PER_BANK:
        raise ValueError(
            f"Segment bank count must be between 2 and "
            f"{HARDWARE_SEGMENTS_PER_BANK}."
        )

    scope.write(f":ACQUIRE:SEGMENTED:COUNT {count}")
    scope.write(f":DIGITIZE {source}")
    scope.query("*OPC?")

    acquired = query_int(scope, ":WAVEFORM:SEGMENTED:COUNT?")
    if acquired is None or acquired <= 0:
        raise RuntimeError(
            "The oscilloscope reported no acquired memory segments."
        )
    return min(acquired, count)


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


def estimate_integration_window(
    time_s: np.ndarray,
    average_voltage_v: np.ndarray,
    pulse_polarity: str,
) -> dict[str, Any]:
    """Suggest an integration window from the average acquired pulse."""

    pretrigger = time_s < 0.0
    if np.count_nonzero(pretrigger) < 4:
        return {}

    baseline_v = float(np.mean(average_voltage_v[pretrigger]))
    baseline_noise_v = float(np.std(average_voltage_v[pretrigger]))
    polarity_sign = -1.0 if pulse_polarity == "NEGATIVE" else 1.0
    pulse_v = polarity_sign * (average_voltage_v - baseline_v)
    peak_index = int(np.argmax(pulse_v))
    peak_v = float(pulse_v[peak_index])
    if not np.isfinite(peak_v) or peak_v <= 0:
        return {}

    threshold_v = max(0.05 * peak_v, 3.0 * baseline_noise_v)
    above = pulse_v >= threshold_v
    start = peak_index
    stop = peak_index
    while start > 0 and above[start - 1]:
        start -= 1
    while stop + 1 < above.size and above[stop + 1]:
        stop += 1

    # Add a small margin to include the pulse tails.
    margin = max(1, int(round(0.02 * time_s.size)))
    start = max(0, start - margin)
    stop = min(time_s.size - 1, stop + margin)

    return {
        "method": (
            "contiguous average-pulse region above max(5% peak, "
            "3x pretrigger noise), with 2% record margin"
        ),
        "baseline_v": baseline_v,
        "baseline_noise_v": baseline_noise_v,
        "average_peak_amplitude_v": peak_v,
        "threshold_v": threshold_v,
        "start_s": float(time_s[start]),
        "stop_s": float(time_s[stop]),
        "start_us": float(time_s[start] * 1.0e6),
        "stop_us": float(time_s[stop] * 1.0e6),
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


def print_progress(
    completed: int,
    target_events: int,
    duration_s: float,
    started_at: float,
) -> None:
    elapsed_s = time.monotonic() - started_at
    fraction = min(completed / target_events, 1.0)
    filled = round(PROGRESS_BAR_WIDTH * fraction)
    bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    print(
        f"\r[{bar}] {fraction * 100:6.2f}% "
        f"({completed}/{target_events} events) "
        f"elapsed {format_duration(elapsed_s)} "
        f"max-time left {format_duration(max(0.0, duration_s - elapsed_s))}",
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
    target_events: int,
    waveform_length_ns: float | None,
    pretrigger_ns: float | None,
    trigger_level_v: float | None,
    pulse_polarity: str,
    csv_file: Path,
    root_file: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int, int, dict[str, Any]]:
    if duration_minutes <= 0:
        raise ValueError("--minutes must be greater than zero.")
    if target_events <= 0:
        raise ValueError("--events must be greater than zero.")

    configure_acquisition(
        scope,
        source,
        waveform_length_ns=waveform_length_ns,
        pretrigger_ns=pretrigger_ns,
        trigger_level_v=trigger_level_v,
        pulse_polarity=pulse_polarity,
    )
    configure_waveform_transfer(scope, source)
    trigger = read_trigger_settings(scope)
    run_settings = {
        "channel": read_channel_settings(scope, source),
        "transfer": read_transfer_settings(scope),
        "acquisition": read_acquisition_settings(scope),
        "recommended_acquisition": {
            "acquisition_type": ACQUISITION_TYPE,
            "acquisition_mode": "SEGMENTED",
            "target_events": target_events,
            "logical_segmented_batch_events": (
                LOGICAL_SEGMENTED_BATCH_EVENTS
            ),
            "hardware_segments_per_bank": (
                HARDWARE_SEGMENTS_PER_BANK
            ),
            "window_source": (
                "oscilloscope"
                if waveform_length_ns is None
                else "command_line_override"
            ),
            "requested_waveform_length_ns": waveform_length_ns,
            "requested_pretrigger_ns": pretrigger_ns,
            "trigger_level_source": (
                "oscilloscope"
                if trigger_level_v is None
                else "command_line_override"
            ),
            "requested_trigger_level_v": trigger_level_v,
            "actual_trigger_level_v": trigger.get("level_v"),
            "pulse_polarity": pulse_polarity,
        },
    }

    root_branches: dict[str, np.ndarray] = {}
    first_arrays: dict[str, np.ndarray] | None = None
    scale: dict[str, Any] | None = None
    voltage_sum_v: np.ndarray | None = None
    points_saved = 0
    hardware_banks_completed = 0
    logical_batches_completed = 0
    started_monotonic = time.monotonic()
    started_epoch = time.time()
    duration_s = duration_minutes * 60.0
    interrupted = False

    try:
        while (
            len(root_branches) < target_events
            and time.monotonic() - started_monotonic < duration_s
        ):
            logical_target = min(
                LOGICAL_SEGMENTED_BATCH_EVENTS,
                target_events - len(root_branches),
            )
            logical_collected = 0

            while (
                logical_collected < logical_target
                and time.monotonic() - started_monotonic < duration_s
            ):
                needed = logical_target - logical_collected
                bank_count = max(
                    2,
                    min(HARDWARE_SEGMENTS_PER_BANK, needed),
                )
                acquired = acquire_segment_bank(
                    scope, source, bank_count
                )
                hardware_banks_completed += 1

                configure_waveform_transfer(scope, source)
                bank_scale: dict[str, Any] | None = None

                for segment_index in range(1, acquired + 1):
                    if len(root_branches) >= target_events:
                        break

                    scope.write(
                        f":ACQUIRE:SEGMENTED:INDEX {segment_index}"
                    )
                    if bank_scale is None:
                        bank_scale = read_scale(scope)

                    arrays = waveform_arrays(
                        read_raw_waveform(scope),
                        bank_scale,
                    )
                    if first_arrays is None:
                        first_arrays = arrays
                        scale = bank_scale
                        points_saved = len(arrays["voltage_v"])
                        voltage_sum_v = np.zeros(
                            points_saved, dtype=np.float64
                        )
                    elif len(arrays["voltage_v"]) != points_saved:
                        raise ValueError(
                            "ROOT output expects each waveform to "
                            "have the same length."
                        )

                    event_index = len(root_branches)
                    voltage = arrays["voltage_v"]
                    root_branches[
                        f"waveform_{event_index:05d}"
                    ] = voltage
                    if voltage_sum_v is None:
                        raise RuntimeError(
                            "Average-waveform accumulator was not initialized."
                        )
                    voltage_sum_v += voltage
                    logical_collected += 1

                print_progress(
                    len(root_branches),
                    target_events,
                    duration_s,
                    started_monotonic,
                )

            if logical_collected == logical_target:
                logical_batches_completed += 1
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
        "target_events": target_events,
        "events_acquired": len(root_branches),
        "completed_target_events": len(root_branches) >= target_events,
        "maximum_duration_reached": (
            not interrupted
            and len(root_branches) < target_events
            and actual_duration_s >= duration_s
        ),
        "logical_batches_completed": logical_batches_completed,
        "hardware_banks_completed": hardware_banks_completed,
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
        raise RuntimeError(
            "No waveforms were collected during the requested time."
        )

    if first_arrays is None or scale is None or voltage_sum_v is None:
        raise RuntimeError("No first waveform is available to save.")

    average_voltage_v = voltage_sum_v / len(root_branches)
    integration_window = estimate_integration_window(
        first_arrays["time_s"],
        average_voltage_v,
        pulse_polarity,
    )
    run_settings["average_pulse"] = {
        "time_s": first_arrays["time_s"].tolist(),
        "voltage_v": average_voltage_v.tolist(),
        "suggested_integration_window": integration_window,
    }
    actual_start_s = float(first_arrays["time_s"][0])
    actual_stop_s = float(first_arrays["time_s"][-1])
    actual_length_s = float(
        actual_stop_s - actual_start_s + scale["xincr"]
    )
    actual_pretrigger_s = float(max(0.0, -actual_start_s))
    run_settings["actual_record"] = {
        "start_s": actual_start_s,
        "stop_s": actual_stop_s,
        "length_s": actual_length_s,
        "pretrigger_s": actual_pretrigger_s,
        "length_ns": actual_length_s * 1.0e9,
        "pretrigger_ns": actual_pretrigger_s * 1.0e9,
    }

    actual_length_ns = actual_length_s * 1.0e9
    actual_pretrigger_ns = actual_pretrigger_s * 1.0e9
    print(
        "Acquired scope window: "
        f"{actual_start_s * 1.0e9:.6g} to "
        f"{actual_stop_s * 1.0e9:.6g} ns "
        f"({actual_length_ns:.6g} ns record, "
        f"{actual_pretrigger_ns:.6g} ns pre-trigger)."
    )
    if not 200.0 <= actual_length_ns <= 500.0:
        print(
            "Warning: the scope waveform length is outside the recommended "
            "200 to 500 ns range.",
            file=sys.stderr,
        )
    if not 50.0 <= actual_pretrigger_ns <= 100.0:
        print(
            "Warning: the scope pre-trigger interval is outside the "
            "recommended 50 to 100 ns range.",
            file=sys.stderr,
        )

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
    run_settings["acquisition_after_run"] = read_acquisition_settings(
        scope
    )
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
    actual_record = run_settings.get("actual_record", {})
    if "start_s" in actual_record and "stop_s" in actual_record:
        metadata["display_time_window_s"] = [
            actual_record["start_s"],
            actual_record["stop_s"],
        ]
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    csv_file, root_file, metadata_file = output_paths(args.output_dir, args.label)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trigger_level_v = trigger_level_from_args(args)

    print("Recommended PMT acquisition")
    print(f"  target events: {args.events}")
    print(
        "  segmented batching: "
        f"{LOGICAL_SEGMENTED_BATCH_EVENTS} logical events "
        f"as banks of {HARDWARE_SEGMENTS_PER_BANK}"
    )
    if args.waveform_ns is None:
        print(
            "  record: use the horizontal window currently configured "
            "on the oscilloscope"
        )
    else:
        print(
            f"  record override: {args.waveform_ns:g} ns total, "
            f"{args.pretrigger_ns:g} ns pre-trigger"
        )
    if trigger_level_v is None:
        print(
            "  trigger: use the level currently configured on the "
            f"oscilloscope ({args.pulse_polarity.lower()} edge)"
        )
    else:
        print(
            f"  trigger override: {args.pulse_polarity.lower()} edge at "
            f"{trigger_level_v:.6g} V"
        )
    if args.high_voltage_kv is not None:
        print(f"  PMT high voltage: {args.high_voltage_kv:g} kV")
    else:
        print(
            "  Warning: no --high-voltage-kv was supplied; "
            "record the HV in the run label.",
            file=sys.stderr,
        )

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
                args.events,
                args.waveform_ns,
                args.pretrigger_ns,
                trigger_level_v,
                args.pulse_polarity,
                csv_file,
                root_file,
            )
            run_settings["recommended_acquisition"].update(
                {
                    "high_voltage_kv": args.high_voltage_kv,
                    "spe_amplitude_v": args.spe_amplitude_v,
                    "trigger_fraction_of_spe": (
                        args.trigger_fraction
                        if args.spe_amplitude_v is not None
                        else None
                    ),
                    "baseline_v": args.baseline_v,
                }
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
    integration_window = (
        run_settings.get("average_pulse", {})
        .get("suggested_integration_window", {})
    )
    if integration_window:
        print(
            "Suggested integration window from average pulse: "
            f"{integration_window['start_us']:.6g} to "
            f"{integration_window['stop_us']:.6g} us"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
