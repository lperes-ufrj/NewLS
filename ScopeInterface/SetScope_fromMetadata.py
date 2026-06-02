#!/usr/bin/env python3
"""Restore a Tektronix MDO3034 setup from one saved metadata JSON file."""

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

# Edit this path when you want to reproduce another acquisition setup.
METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "PMT_Calibration"
    / "metadata_hybrid_pmt_calibration_1p1kv_10k_hires_0601.json"
)

EXPECTED_VENDOR = "TEKTRONIX"
EXPECTED_MODEL = "MDO3034"
DEFAULT_VISA_RESOURCE = "USB0::1689::1032::C053047::0::INSTR"
TIMEOUT_MS = 10_000


def configure_scope_connection(scope: Any) -> None:
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


def load_metadata(metadata_file: Path) -> dict[str, Any]:
    if not metadata_file.exists():
        raise FileNotFoundError(metadata_file)
    return json.loads(metadata_file.read_text())


def metadata_visa_resource(metadata: dict[str, Any]) -> str:
    return str(metadata.get("visa_resource") or DEFAULT_VISA_RESOURCE)


def write_if_present(scope: Any, command: str, value: Any) -> None:
    if value is None:
        return
    scope.write(f"{command} {value}")


def select_only_channel(scope: Any, source: str) -> None:
    for channel in ("CH1", "CH2", "CH3", "CH4"):
        scope.write(f"SELECT:{channel} {1 if channel == source else 0}")


def restore_horizontal(scope: Any, metadata: dict[str, Any]) -> None:
    scale = metadata.get("scale", {})
    write_if_present(
        scope,
        "HORIZONTAL:RECORDLENGTH",
        as_int(scale.get("record_length") or scale.get("waveform_points")),
    )
    write_if_present(scope, "HORIZONTAL:SCALE", scale.get("horizontal_scale_s_per_div"))
    write_if_present(scope, "HORIZONTAL:DELAY:MODE", scale.get("horizontal_delay_mode"))
    write_if_present(scope, "HORIZONTAL:DELAY:TIME", scale.get("horizontal_delay_time_s"))


def restore_channel(scope: Any, metadata: dict[str, Any]) -> str:
    channel = metadata.get("scope_settings", {}).get("channel", {})
    source = str(channel.get("source") or metadata.get("waveform_source") or "CH1").upper()

    select_only_channel(scope, source)
    write_if_present(scope, f"{source}:SCALE", channel.get("scale_v_per_div"))
    write_if_present(scope, f"{source}:POSITION", channel.get("position_div"))
    write_if_present(scope, f"{source}:OFFSET", channel.get("offset_v"))
    write_if_present(scope, f"{source}:COUPLING", channel.get("coupling"))
    write_if_present(scope, f"{source}:TERMINATION", as_int(channel.get("termination_ohm")))
    write_if_present(scope, f"{source}:BANDWIDTH", channel.get("bandwidth"))
    write_if_present(scope, f"{source}:INVERT", channel.get("invert"))

    return source


def restore_trigger(scope: Any, metadata: dict[str, Any], source: str) -> None:
    trigger = metadata.get("trigger", {})
    write_if_present(scope, "TRIGGER:A:TYPE", trigger.get("type"))
    write_if_present(scope, "TRIGGER:A:EDGE:SOURCE", trigger.get("source") or source)
    write_if_present(scope, "TRIGGER:A:EDGE:SLOPE", trigger.get("slope"))
    write_if_present(scope, "TRIGGER:A:LEVEL", trigger.get("level_v"))


def restore_acquisition(scope: Any, metadata: dict[str, Any]) -> None:
    acquisition = metadata.get("scope_settings", {}).get("acquisition", {})
    scale = metadata.get("scale", {})
    write_if_present(scope, "ACQUIRE:MODE", acquisition.get("mode") or scale.get("acquisition_mode"))
    write_if_present(scope, "ACQUIRE:NUMAVG", as_int(acquisition.get("num_average")))
    write_if_present(scope, "ACQUIRE:STOPAFTER", acquisition.get("stop_after"))


def restore_transfer(scope: Any, metadata: dict[str, Any], source: str) -> None:
    transfer = metadata.get("scope_settings", {}).get("transfer", {})
    scale = metadata.get("scale", {})

    scope.write(f"DATA:SOURCE {transfer.get('data_source') or source}")
    write_if_present(scope, "DATA:START", as_int(transfer.get("data_start") or 1))
    write_if_present(
        scope,
        "DATA:STOP",
        as_int(
            transfer.get("data_stop")
            or scale.get("record_length")
            or scale.get("waveform_points")
        ),
    )
    write_if_present(scope, "DATA:ENC", transfer.get("data_encoding"))
    write_if_present(
        scope,
        "DATA:WIDTH",
        as_int(transfer.get("data_width_bytes") or scale.get("transfer_width_bytes")),
    )


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(float(value))


def query_or_unknown(scope: Any, command: str) -> str:
    try:
        return scope.query(command).strip()
    except VisaError:
        return "?"


def print_summary(scope: Any, source: str) -> None:
    print("Applied scope setup:")
    print(f"  source              = {source}")
    print(f"  horizontal scale    = {query_or_unknown(scope, 'HORIZONTAL:SCALE?')} s/div")
    print(f"  horizontal delay    = {query_or_unknown(scope, 'HORIZONTAL:DELAY:TIME?')} s")
    print(f"  horizontal delay on = {query_or_unknown(scope, 'HORIZONTAL:DELAY:MODE?')}")
    print(f"  record length       = {query_or_unknown(scope, 'HORIZONTAL:RECORDLENGTH?')}")
    print(f"  channel scale       = {query_or_unknown(scope, f'{source}:SCALE?')} V/div")
    print(f"  channel position    = {query_or_unknown(scope, f'{source}:POSITION?')} div")
    print(f"  trigger source      = {query_or_unknown(scope, 'TRIGGER:A:EDGE:SOURCE?')}")
    print(f"  trigger slope       = {query_or_unknown(scope, 'TRIGGER:A:EDGE:SLOPE?')}")
    print(f"  trigger level       = {query_or_unknown(scope, 'TRIGGER:A:LEVEL?')} V")
    print(f"  acquisition mode    = {query_or_unknown(scope, 'ACQUIRE:MODE?')}")
    print(f"  acquisition state   = {query_or_unknown(scope, 'ACQUIRE:STATE?')}")
    print(f"  data source         = {query_or_unknown(scope, 'DATA:SOURCE?')}")
    print(f"  data width          = {query_or_unknown(scope, 'DATA:WIDTH?')} byte(s)")


def restore_scope_from_metadata(scope: Any, metadata: dict[str, Any]) -> str:
    scope.write("ACQUIRE:STATE STOP")
    source = restore_channel(scope, metadata)
    restore_horizontal(scope, metadata)
    restore_trigger(scope, metadata, source)
    restore_acquisition(scope, metadata)
    restore_transfer(scope, metadata, source)
    scope.write("*WAI")
    scope.write("ACQUIRE:STATE RUN")
    return source


def main() -> int:
    try:
        metadata = load_metadata(METADATA_FILE)
        visa_resource = metadata_visa_resource(metadata)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read metadata file {METADATA_FILE}: {exc}", file=sys.stderr)
        return 1

    try:
        manager = pyvisa.ResourceManager()
    except (OSError, ValueError, VisaError) as exc:
        print(f"Could not initialize a VISA resource manager: {exc}", file=sys.stderr)
        return 2

    try:
        with manager.open_resource(visa_resource) as scope:
            configure_scope_connection(scope)
            idn = scope.query("*IDN?").strip()
            print(f"{visa_resource}: {idn}")
            if not is_mdo3034(idn):
                print(
                    "The responding instrument is not a Tektronix MDO3034.",
                    file=sys.stderr,
                )
                return 1

            source = restore_scope_from_metadata(scope, metadata)
            print_summary(scope, source)
    except (RuntimeError, ValueError, VisaError) as exc:
        print(f"Could not restore setup from {METADATA_FILE}: {exc}", file=sys.stderr)
        return 1
    finally:
        manager.close()

    print(f"OK: restored setup from {METADATA_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
