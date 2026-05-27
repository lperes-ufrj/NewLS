import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot


BASE_DIR = Path(__file__).resolve().parent
ROOT_FILE = BASE_DIR / "waveforms_hybrid_pmt_calibration_1p1kv.root"
METADATA_FILE = BASE_DIR / "metadata_hybrid_pmt_calibration_1p1kv.json"
WAVEFORM_INDICES = [11,12,13,14,15,16,17]


def read_metadata():
    if not METADATA_FILE.exists():
        return {}
    return json.loads(METADATA_FILE.read_text())


def display_time_window_us(metadata):
    if "display_time_window_s" in metadata:
        return [time_s * 1e6 for time_s in metadata["display_time_window_s"]]

    scale = metadata.get("scale", {})
    if "horizontal_scale_s_per_div" not in scale:
        return None

    center_s = scale.get("horizontal_delay_time_s", 0.0)
    if not scale.get("horizontal_delay_mode", 0):
        center_s = 0.0

    half_width_s = 5.0 * scale["horizontal_scale_s_per_div"]
    return [(center_s - half_width_s) * 1e6, (center_s + half_width_s) * 1e6]


def open_waveform_tree():
    if not ROOT_FILE.exists():
        raise FileNotFoundError(ROOT_FILE)

    root_file = uproot.open(ROOT_FILE)
    if "waveforms" not in root_file:
        root_file.close()
        raise RuntimeError("ROOT file does not contain a 'waveforms' tree.")
    return root_file, root_file["waveforms"]


def read_root_waveform(tree, waveform_index):
    branch_name = f"waveform_{waveform_index:05d}"
    if branch_name not in tree:
        raise IndexError(f"ROOT file does not contain branch {branch_name!r}.")

    # These files have one basket per waveform branch. Reading the basket data
    # directly avoids loading thousands of sibling branches.
    basket = tree[branch_name].basket(0)
    return np.frombuffer(basket.data, dtype=">f8").astype(float)


def time_axis_us(points, time_window_us):
    if time_window_us is None:
        return np.arange(points)
    return np.linspace(time_window_us[0], time_window_us[1], points)


def main():
    metadata = read_metadata()
    time_window_us = display_time_window_us(metadata)
    root_file, tree = open_waveform_tree()

    fig, ax = plt.subplots(figsize=(10, 5))

    try:
        for waveform_index in WAVEFORM_INDICES:
            voltage = read_root_waveform(tree, waveform_index)
            time_us = time_axis_us(len(voltage), time_window_us)
            ax.plot(
                time_us,
                voltage,
                linewidth=1,
                label=f"waveform_{waveform_index:05d}",
            )
    finally:
        root_file.close()

    if time_window_us is None:
        ax.set_xlabel("Sample")
    else:
        ax.set_xlim(time_window_us)
        ax.set_xlabel("Time (us)")
    ax.set_ylabel("Voltage (V)")

    if len(WAVEFORM_INDICES) > 1:
        ax.legend()

    ax.set_title("Tektronix MDO3034 Waveform")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
