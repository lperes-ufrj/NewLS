import json
from pathlib import Path

import numpy as np
import uproot


def read_metadata(metadata_file):
    metadata_file = Path(metadata_file)
    if not metadata_file.exists():
        return {}
    return json.loads(metadata_file.read_text())


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


def open_waveform_tree(root_file, tree_name="waveforms"):
    root_file = Path(root_file)
    if not root_file.exists():
        raise FileNotFoundError(root_file)

    root_handle = uproot.open(root_file)
    if tree_name not in root_handle:
        root_handle.close()
        raise RuntimeError(f"ROOT file does not contain a {tree_name!r} tree.")
    return root_handle, root_handle[tree_name]


def read_root_waveform(tree, waveform_index):
    branch_name = waveform_branch_name(waveform_index)
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


def waveform_branch_name(waveform_index):
    return f"waveform_{waveform_index:05d}"


def read_waveforms(root_file, waveform_indices, tree_name="waveforms"):
    root_handle, tree = open_waveform_tree(root_file, tree_name)

    try:
        return {
            waveform_index: read_root_waveform(tree, waveform_index)
            for waveform_index in waveform_indices
        }
    finally:
        root_handle.close()


def load_waveforms(root_file, waveform_indices, metadata_file=None, tree_name="waveforms"):
    metadata = read_metadata(metadata_file) if metadata_file is not None else {}
    time_window_us = display_time_window_us(metadata)
    waveforms = read_waveforms(root_file, waveform_indices, tree_name)

    if waveforms:
        first_waveform = next(iter(waveforms.values()))
        time_us = time_axis_us(len(first_waveform), time_window_us)
    else:
        time_us = np.array([])

    return time_us, waveforms, time_window_us
