from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CSV_FILE = Path("mdo3034_ch1_waveforms_1p2_kV/waveform_00099.csv")


def main():
    waveform = pd.read_csv(CSV_FILE)

    fig, ax = plt.subplots(figsize=(10, 5))
    if {"time_s", "voltage_v"}.issubset(waveform.columns):
        ax.plot(waveform["time_s"] * 1e6, waveform["voltage_v"], linewidth=1)
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Voltage (V)")
    else:
        ax.plot(waveform["raw_digitizing_level"], linewidth=1)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Raw digitizing level")

    ax.set_title("Tektronix MDO3034 Waveform")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
