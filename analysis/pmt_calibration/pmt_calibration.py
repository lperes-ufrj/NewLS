import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.ReadWaveForms import load_waveforms, waveform_branch_name

DATA_DIR = PROJECT_ROOT / "data" / "PMT_Calibration"
ROOT_FILE = DATA_DIR / "waveforms_hybrid_pmt_calibration_1p2kv.root"
METADATA_FILE = DATA_DIR / "metadata_hybrid_pmt_calibration_1p2kv.json"
WAVEFORM_INDICES = np.linspace(0, 2999, 200, dtype=int)

def main():
    time_us, waveforms, time_window_us = load_waveforms(
        ROOT_FILE,
        WAVEFORM_INDICES,
        METADATA_FILE,
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    for waveform_index, voltage in waveforms.items():
        if np.any(np.array(voltage) < 0.06): continue 
        ax.plot(
            time_us,
            voltage,
            'o',
            label=waveform_branch_name(waveform_index),
        )

    if time_window_us is None:
        ax.set_xlabel("Sample")
    else:
        ax.set_xlim([-0.2,0.2])
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
