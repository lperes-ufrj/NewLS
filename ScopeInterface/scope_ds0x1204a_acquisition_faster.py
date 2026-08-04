#!/usr/bin/env python3
"""Acquire DSOX1204A waveforms and write the same ROOT schema faster.

This is a minimally changed variant of scope_ds0x1204a_acquisition.py. It
reuses that module's acquisition, CSV, metadata, and command-line behavior,
but uses fast ROOT compression and avoids an intermediate Python sample list.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import uproot

import scope_ds0x1204a_acquisition as acquisition


# Compression changes only the physical representation, not the tree schema,
# branch types, or values. LZ4 is faster than the default ZLIB compression.
ROOT_COMPRESSION = uproot.LZ4(1)


def read_raw_waveform(scope: Any) -> np.ndarray:
    """Receive samples directly into NumPy instead of building a Python list."""

    return scope.query_binary_values(
        ":WAVEFORM:DATA?",
        datatype="H",
        is_big_endian=False,
        container=np.array,
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

    acquisition.configure_acquisition(scope)
    acquisition.configure_waveform_transfer(scope, source)
    trigger = acquisition.read_trigger_settings(scope)
    run_settings = {
        "channel": acquisition.read_channel_settings(scope, source),
        "transfer": acquisition.read_transfer_settings(scope),
        "acquisition": acquisition.read_acquisition_settings(scope),
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
            acquisition.wait_for_triggered_acquisition(scope, source)
            acquisition.configure_waveform_transfer(scope, source)
            current_scale = acquisition.read_scale(scope)
            arrays = acquisition.waveform_arrays(
                read_raw_waveform(scope), current_scale
            )
            if first_arrays is None:
                first_arrays = arrays
                scale = current_scale
                points_saved = len(arrays["voltage_v"])
            elif len(arrays["voltage_v"]) != points_saved:
                raise ValueError(
                    "ROOT output expects each saved waveform to have the same length."
                )
            root_branches[f"waveform_{index:05d}"] = arrays["voltage_v"]
            acquisition.print_progress(index + 1, duration_s, started_monotonic)
    except KeyboardInterrupt:
        interrupted = True

    print()
    actual_duration_s = time.time() - started_epoch
    timing = {
        "requested_duration_minutes": duration_minutes,
        "requested_duration_s": duration_s,
        "actual_duration_s": actual_duration_s,
        "actual_duration_minutes": actual_duration_s / 60.0,
        "started_at": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(started_epoch)
        ),
        "ended_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "completed_requested_duration": not interrupted,
        "interrupted": interrupted,
    }
    if interrupted:
        print(
            "Acquisition interrupted after "
            f"{acquisition.format_duration(actual_duration_s)} with "
            f"{len(root_branches)} waveforms."
        )
        if not root_branches:
            raise RuntimeError("No waveforms were collected before the interruption.")
        if not acquisition.prompt_yes_no("Save the waveforms collected so far?"):
            raise RuntimeError("Acquisition interrupted; collected data was not saved.")
    elif not root_branches:
        raise RuntimeError("No waveforms were collected during the requested time.")

    if first_arrays is None or scale is None:
        raise RuntimeError("No first waveform is available to save.")
    acquisition.write_waveform_csv(csv_file, first_arrays)

    print(f"Writing ROOT file with {len(root_branches)} waveform branches...")
    root_write_started = time.perf_counter()
    with uproot.recreate(root_file, compression=ROOT_COMPRESSION) as root:
        root.mktree("waveforms", root_branches)
    print(
        "ROOT file written in "
        f"{acquisition.format_duration(time.perf_counter() - root_write_started)}."
    )

    run_settings["effective_resolution"] = {
        "yincr_v_per_raw_count": scale["yincr"],
        "transfer_format": scale.get(
            "transfer_format", acquisition.TRANSFER_FORMAT
        ),
    }
    return (
        scale,
        trigger,
        run_settings,
        points_saved,
        len(root_branches),
        timing,
    )


def main() -> int:
    acquisition.acquire_displayed_waveforms = acquire_displayed_waveforms
    return acquisition.main()


if __name__ == "__main__":
    raise SystemExit(main())
