#!/usr/bin/env python3
"""Restore saved scope metadata to a connected Tektronix or Keysight scope."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    import pyvisa
    from pyvisa.errors import Error as VisaError
except ImportError as exc:
    raise SystemExit(
        "PyVISA is required. Install it and a VISA backend before running this script."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Change only this path to select the setup that should be reproduced.
METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "RadioSources_Calibration"
    / "metadata_hybrid_pmt_1p2kv_cs137_keysight_30min_1_1_normal.json"
)
TIMEOUT_MS = 10_000
SUPPORTED_SCOPES = {
    ("TEKTRONIX", "MDO3034"): "tektronix",
    ("KEYSIGHT TECHNOLOGIES", "DSOX1204A"): "keysight",
    ("AGILENT TECHNOLOGIES", "DSOX1204A"): "keysight",
}


def configure_scope_connection(scope: Any) -> None:
    scope.timeout = TIMEOUT_MS
    scope.read_termination = "\n"
    scope.write_termination = "\n"


def identify_scope(idn: str) -> str | None:
    fields = [field.strip().upper() for field in idn.split(",")]
    if len(fields) < 2:
        return None
    return SUPPORTED_SCOPES.get((fields[0], fields[1]))


def load_metadata(metadata_file: Path) -> dict[str, Any]:
    if not metadata_file.exists():
        raise FileNotFoundError(metadata_file)
    metadata = json.loads(metadata_file.read_text())
    metadata_scope = identify_scope(str(metadata.get("idn", "")))
    if metadata_scope is None:
        raise ValueError(
            "The metadata does not identify a supported Tektronix MDO3034 "
            "or Keysight DSOX1204A."
        )
    return metadata


def find_connected_scope(manager: Any) -> tuple[Any, str, str, str]:
    """Return the only connected supported scope and its identity."""

    matches: list[tuple[Any, str, str, str]] = []
    for resource in manager.list_resources():
        scope = None
        try:
            scope = manager.open_resource(resource)
            configure_scope_connection(scope)
            idn = scope.query("*IDN?").strip()
            scope_kind = identify_scope(idn)
            if scope_kind is not None:
                matches.append((scope, scope_kind, resource, idn))
                scope = None
        except (OSError, ValueError, VisaError):
            pass
        finally:
            if scope is not None:
                scope.close()

    if not matches:
        raise RuntimeError(
            "No supported Tektronix MDO3034 or Keysight DSOX1204A was found."
        )
    if len(matches) > 1:
        for scope, _, _, _ in matches:
            scope.close()
        resources = ", ".join(match[2] for match in matches)
        raise RuntimeError(
            f"More than one supported scope is connected ({resources}). "
            "Disconnect one scope and run the script again."
        )
    return matches[0]


def write_if_present(scope: Any, command: str, value: Any) -> None:
    if value is not None:
        scope.write(f"{command} {value}")


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(float(value))


def channel_number(source: Any) -> int:
    text = str(source or "1").upper()
    for prefix in ("CHANNEL", "CHAN", "CH"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    number = int(text)
    if number not in (1, 2, 3, 4):
        raise ValueError(f"Unsupported analog waveform source: {source!r}")
    return number


def target_source(source: Any, target_scope: str) -> str:
    number = channel_number(source)
    return f"CH{number}" if target_scope == "tektronix" else f"CHAN{number}"


def mapped_slope(slope: Any, target_scope: str) -> Any:
    if slope is None:
        return None
    value = str(slope).upper()
    if target_scope == "tektronix":
        return {
            "ALT": "EITH",
            "EITHER": "EITH",
            "NEG": "FALL",
            "NEGATIVE": "FALL",
            "POS": "RISE",
            "POSITIVE": "RISE",
        }.get(value, value)
    return {
        "EITH": "ALT",
        "EITHER": "ALT",
        "FALL": "NEG",
        "FALLING": "NEG",
        "RISE": "POS",
        "RISING": "POS",
    }.get(value, value)


def mapped_acquisition_type(value: Any, target_scope: str) -> Any:
    if value is None:
        return None
    mode = str(value).upper()
    if target_scope == "tektronix":
        return {"HRES": "HIR", "NORMAL": "SAMPL"}.get(mode, mode)
    return {"HIR": "HRES", "SAMPL": "NORMAL", "SAMPLE": "NORMAL"}.get(mode, mode)


def restore_tektronix(scope: Any, metadata: dict[str, Any]) -> str:
    scale = metadata.get("scale", {})
    settings = metadata.get("scope_settings", {})
    channel = settings.get("channel", {})
    trigger = metadata.get("trigger", {})
    acquisition = settings.get("acquisition", {})
    transfer = settings.get("transfer", {})
    source = target_source(
        channel.get("source") or metadata.get("waveform_source"), "tektronix"
    )
    trigger_source = target_source(trigger.get("source") or source, "tektronix")

    scope.write("ACQUIRE:STATE STOP")
    for candidate in ("CH1", "CH2", "CH3", "CH4"):
        scope.write(f"SELECT:{candidate} {1 if candidate == source else 0}")

    write_if_present(scope, f"{source}:SCALE", channel.get("scale_v_per_div"))
    write_if_present(scope, f"{source}:POSITION", channel.get("position_div"))
    write_if_present(scope, f"{source}:OFFSET", channel.get("offset_v"))
    write_if_present(scope, f"{source}:COUPLING", channel.get("coupling"))
    write_if_present(
        scope,
        f"{source}:TERMINATION",
        as_int(channel.get("termination_ohm") or channel.get("impedance_ohm")),
    )
    bandwidth = channel.get("bandwidth")
    if bandwidth is None and str(channel.get("bandwidth_limit")) == "0":
        bandwidth = "FULL"
    write_if_present(scope, f"{source}:BANDWIDTH", bandwidth)
    write_if_present(scope, f"{source}:INVERT", channel.get("invert"))

    write_if_present(
        scope, "HORIZONTAL:SCALE", scale.get("horizontal_scale_s_per_div")
    )
    delay_time = scale.get("horizontal_delay_time_s")
    if delay_time is None:
        delay_time = scale.get("horizontal_position_s")
    write_if_present(scope, "HORIZONTAL:DELAY:TIME", delay_time)
    write_if_present(
        scope, "HORIZONTAL:DELAY:MODE", scale.get("horizontal_delay_mode")
    )
    write_if_present(
        scope,
        "HORIZONTAL:RECORDLENGTH",
        as_int(scale.get("record_length") or scale.get("waveform_points")),
    )

    scope.write("TRIGGER:A:TYPE EDGE")
    write_if_present(scope, "TRIGGER:A:EDGE:SOURCE", trigger_source)
    write_if_present(
        scope, "TRIGGER:A:EDGE:SLOPE", mapped_slope(trigger.get("slope"), "tektronix")
    )
    write_if_present(scope, "TRIGGER:A:LEVEL", trigger.get("level_v"))

    acquisition_type = (
        acquisition.get("mode")
        or acquisition.get("type")
        or scale.get("acquisition_mode")
        or scale.get("acquisition_type")
    )
    write_if_present(
        scope,
        "ACQUIRE:MODE",
        mapped_acquisition_type(acquisition_type, "tektronix"),
    )
    write_if_present(scope, "ACQUIRE:NUMAVG", as_int(acquisition.get("num_average")))

    scope.write(f"DATA:SOURCE {source}")
    scope.write("DATA:START 1")
    write_if_present(
        scope,
        "DATA:STOP",
        as_int(
            transfer.get("data_stop")
            or scale.get("record_length")
            or scale.get("waveform_points")
        ),
    )
    width = as_int(
        transfer.get("data_width_bytes")
        or (2 if str(transfer.get("format", "")).upper() == "WORD" else None)
        or scale.get("transfer_width_bytes")
    )
    scope.write("DATA:ENC RIB")
    write_if_present(scope, "DATA:WIDTH", width)
    scope.write("*WAI")
    scope.write("ACQUIRE:STATE RUN")
    return source


def restore_keysight(scope: Any, metadata: dict[str, Any]) -> str:
    scale = metadata.get("scale", {})
    settings = metadata.get("scope_settings", {})
    channel = settings.get("channel", {})
    trigger = metadata.get("trigger", {})
    acquisition = settings.get("acquisition", {})
    transfer = settings.get("transfer", {})
    source = target_source(
        channel.get("source") or metadata.get("waveform_source"), "keysight"
    )
    trigger_source = target_source(trigger.get("source") or source, "keysight")

    scope.write(":STOP")
    for candidate in ("CHAN1", "CHAN2", "CHAN3", "CHAN4"):
        scope.write(f":{candidate}:DISPLAY {1 if candidate == source else 0}")

    write_if_present(scope, f":{source}:SCALE", channel.get("scale_v_per_div"))
    write_if_present(scope, f":{source}:OFFSET", channel.get("offset_v"))
    write_if_present(scope, f":{source}:COUPLING", channel.get("coupling"))
    write_if_present(
        scope,
        f":{source}:IMPEDANCE",
        channel.get("impedance_ohm") or channel.get("termination_ohm"),
    )
    write_if_present(scope, f":{source}:BWLIMIT", channel.get("bandwidth_limit"))
    write_if_present(scope, f":{source}:INVERT", channel.get("invert"))
    write_if_present(scope, f":{source}:PROBE", channel.get("probe_ratio"))

    scope.write(":TIMEBASE:MODE MAIN")
    write_if_present(
        scope, ":TIMEBASE:SCALE", scale.get("horizontal_scale_s_per_div")
    )
    position = scale.get("horizontal_position_s")
    if position is None:
        position = scale.get("horizontal_delay_time_s")
    write_if_present(scope, ":TIMEBASE:POSITION", position)
    write_if_present(scope, ":TIMEBASE:REFERENCE", scale.get("horizontal_reference"))

    scope.write(":TRIGGER:MODE EDGE")
    write_if_present(scope, ":TRIGGER:EDGE:SOURCE", trigger_source)
    write_if_present(
        scope, ":TRIGGER:EDGE:SLOPE", mapped_slope(trigger.get("slope"), "keysight")
    )
    write_if_present(scope, ":TRIGGER:EDGE:LEVEL", trigger.get("level_v"))
    write_if_present(scope, ":TRIGGER:SWEEP", trigger.get("sweep"))

    acquisition_type = (
        acquisition.get("type")
        or acquisition.get("mode")
        or scale.get("acquisition_type")
        or scale.get("acquisition_mode")
    )
    write_if_present(
        scope,
        ":ACQUIRE:TYPE",
        mapped_acquisition_type(acquisition_type, "keysight"),
    )
    acquisition_mode = str(acquisition.get("mode") or "").upper()
    if acquisition_mode.startswith("SEGM"):
        scope.write(":ACQUIRE:MODE SEGMENTED")
    write_if_present(scope, ":ACQUIRE:COUNT", as_int(acquisition.get("count")))

    scope.write(f":WAVEFORM:SOURCE {source}")
    width = as_int(
        transfer.get("data_width_bytes") or scale.get("transfer_width_bytes")
    )
    waveform_format = transfer.get("format") or scale.get("transfer_format")
    if waveform_format is None:
        waveform_format = "WORD" if width == 2 else "BYTE"
    write_if_present(scope, ":WAVEFORM:FORMAT", waveform_format)
    byte_order = transfer.get("byte_order") or scale.get("byte_order")
    if byte_order is not None:
        byte_order = {"LSB": "LSBF", "MSB": "MSBF"}.get(
            str(byte_order).upper(), byte_order
        )
    write_if_present(scope, ":WAVEFORM:BYTEORDER", byte_order)
    write_if_present(scope, ":WAVEFORM:UNSIGNED", transfer.get("unsigned"))
    write_if_present(scope, ":WAVEFORM:POINTS:MODE", transfer.get("points_mode"))
    write_if_present(
        scope,
        ":WAVEFORM:POINTS",
        as_int(
            transfer.get("points")
            or transfer.get("data_stop")
            or scale.get("waveform_points")
        ),
    )
    scope.query("*OPC?")
    scope.write(":RUN")
    return source


def query_or_unknown(scope: Any, command: str) -> str:
    try:
        return scope.query(command).strip()
    except VisaError:
        return "?"


def print_summary(scope: Any, scope_kind: str, source: str) -> None:
    print("Applied scope setup:")
    print(f"  connected scope     = {scope_kind}")
    print(f"  source              = {source}")
    if scope_kind == "tektronix":
        queries = (
            ("horizontal scale", "HORIZONTAL:SCALE?"),
            ("channel scale", f"{source}:SCALE?"),
            ("trigger source", "TRIGGER:A:EDGE:SOURCE?"),
            ("trigger slope", "TRIGGER:A:EDGE:SLOPE?"),
            ("trigger level", "TRIGGER:A:LEVEL?"),
            ("acquisition mode", "ACQUIRE:MODE?"),
        )
    else:
        queries = (
            ("horizontal scale", ":TIMEBASE:SCALE?"),
            ("channel scale", f":{source}:SCALE?"),
            ("trigger source", ":TRIGGER:EDGE:SOURCE?"),
            ("trigger slope", ":TRIGGER:EDGE:SLOPE?"),
            ("trigger level", ":TRIGGER:EDGE:LEVEL?"),
            ("acquisition type", ":ACQUIRE:TYPE?"),
        )
    for label, command in queries:
        print(f"  {label:<19} = {query_or_unknown(scope, command)}")


def restore_scope_from_metadata(
    scope: Any, connected_scope: str, metadata: dict[str, Any]
) -> str:
    if connected_scope == "tektronix":
        return restore_tektronix(scope, metadata)
    return restore_keysight(scope, metadata)


def main() -> int:
    try:
        metadata = load_metadata(METADATA_FILE)
        metadata_scope = identify_scope(str(metadata["idn"]))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Could not read metadata file {METADATA_FILE}: {exc}", file=sys.stderr)
        return 1

    try:
        manager = pyvisa.ResourceManager()
    except (OSError, ValueError, VisaError) as exc:
        print(f"Could not initialize a VISA resource manager: {exc}", file=sys.stderr)
        return 2

    scope = None
    try:
        scope, connected_scope, resource, idn = find_connected_scope(manager)
        print(f"Metadata scope:  {metadata_scope} ({metadata['idn']})")
        print(f"Connected scope: {connected_scope} ({idn})")
        print(f"VISA resource:   {resource}")
        source = restore_scope_from_metadata(scope, connected_scope, metadata)
        print_summary(scope, connected_scope, source)
    except (RuntimeError, ValueError, VisaError) as exc:
        print(f"Could not restore setup from {METADATA_FILE}: {exc}", file=sys.stderr)
        return 1
    finally:
        if scope is not None:
            scope.close()
        manager.close()

    print(f"OK: restored setup from {METADATA_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
