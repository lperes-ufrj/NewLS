import sys
from pathlib import Path

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.ReadWaveForms import load_waveforms

DATA_DIR = PROJECT_ROOT / "data" / "RadioSources_Calibration"
ROOT_FILE = DATA_DIR / "waveforms_hybrid_pmt_1p1kv_10k_hires_cs137_0603_aftern2.root"
METADATA_FILE = DATA_DIR / "metadata_hybrid_pmt_1p1kv_10k_hires_cs137_0603_aftern2.json"
WAVEFORM_INDICES = np.arange(10000)
PLOTS_DIR = PROJECT_ROOT / "analysis" / "plots"

BASELINE_WINDOW_MAX_US = -0.05
INTEGRATION_WINDOW_US = (-0.02, 0.03)
SIGNAL_WINDOW_US = (-0.025, 0.05)
OUT_OF_WINDOW_MAX_V = 0.15
MAX_ACCEPTED_HEIGHT_V = 0.15
MIN_ACCEPTED_HEIGHT_V = 0.0


def integrate_waveform(time_us, voltage, integration_mask):
    return np.trapz(voltage[integration_mask] * 1e3, time_us[integration_mask] * 1e3)


def main():
    time_us, waveforms, time_window_us = load_waveforms(
        ROOT_FILE,
        WAVEFORM_INDICES,
        METADATA_FILE,
    )

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{ROOT_FILE.stem}", fontsize=10)
    ax_waveforms = axs[0, 0]
    ax_sum = axs[0, 1]
    ax_integral = axs[1, 0]
    ax_integral_height = axs[1, 1]

    baseline_mask = time_us < BASELINE_WINDOW_MAX_US
    integration_mask = (
        (time_us >= INTEGRATION_WINDOW_US[0])
        & (time_us <= INTEGRATION_WINDOW_US[1])
    )
    outside_signal_mask = (
        (time_us < SIGNAL_WINDOW_US[0])
        | (time_us > SIGNAL_WINDOW_US[1])
    )
    if not np.any(baseline_mask):
        raise RuntimeError(
            "Baseline window is empty. "
            f"Time range is {time_us.min():.6g} to {time_us.max():.6g} us, "
            f"but BASELINE_WINDOW_MAX_US is {BASELINE_WINDOW_MAX_US:g}."
        )
    if not np.any(integration_mask):
        raise RuntimeError(
            "Integration window is empty. "
            f"Time range is {time_us.min():.6g} to {time_us.max():.6g} us, "
            f"but INTEGRATION_WINDOW_US is {INTEGRATION_WINDOW_US}."
        )

    finger_plot = []
    max_height = []
    voltage_sum = np.zeros_like(time_us)
    rejected_out_of_window = 0
    rejected_too_high = 0
    rejected_too_low = 0

    for waveform_index, voltage in waveforms.items():
        baseline = np.mean(voltage[baseline_mask])
        voltage = -(voltage - baseline)
        if np.any(voltage[outside_signal_mask] > OUT_OF_WINDOW_MAX_V):
            rejected_out_of_window += 1
            continue
        height = np.max(voltage)

        if height > MAX_ACCEPTED_HEIGHT_V:
            rejected_too_high += 1
            continue
        if height < MIN_ACCEPTED_HEIGHT_V:
            rejected_too_low += 1
            continue

        voltage_sum += voltage

        #print(f"div: {voltage[5]-voltage[4]} V at {time_us[5]:.3f} us")
        finger_plot.append(integrate_waveform(time_us, voltage, integration_mask))
        max_height.append(height)

        ax_waveforms.plot(time_us, voltage, linewidth=1)

    finger_plot = np.array(finger_plot)
    max_height = np.array(max_height)
    finite_mask = np.isfinite(finger_plot) & np.isfinite(max_height)
    finger_plot = finger_plot[finite_mask]
    max_height = max_height[finite_mask]
    print(
        f"Rejected {rejected_out_of_window} waveforms with voltage "
        f"> {OUT_OF_WINDOW_MAX_V:g} V outside "
        f"{SIGNAL_WINDOW_US[0]:g} to {SIGNAL_WINDOW_US[1]:g} us."
    )
    print(
        f"Accepted {finger_plot.size} waveforms; rejected "
        f"{rejected_too_high} above {MAX_ACCEPTED_HEIGHT_V:g} V and "
        f"{rejected_too_low} below {MIN_ACCEPTED_HEIGHT_V:g} V."
    )

    ax_waveforms.set_title("Tektronix MDO2024 Waveform")
    ax_waveforms.set_ylabel("Voltage (V)")
    if time_window_us is None:
        ax_waveforms.set_xlabel("Sample")
    else:
        ax_waveforms.set_xlim([-0.2, 0.2])
        ax_waveforms.set_xlabel("Time (us)")
    ax_waveforms.axvspan(
        INTEGRATION_WINDOW_US[0],
        INTEGRATION_WINDOW_US[1],
        color="tab:orange",
        alpha=0.2,
        label="Integration window",
    )
    ax_waveforms.grid(True, alpha=0.35)
    ax_waveforms.legend(title=f"Number of Waveforms: {finger_plot.size:.0f}", fontsize=8)

    ax_sum.plot(time_us, voltage_sum, label="Sum of Waveforms")
    ax_sum.fill_between(
        time_us[integration_mask],
        voltage_sum[integration_mask],
        alpha=0.25,
        color="tab:orange",
        label="Integrated area",
    )
    ax_sum.set_title("Sum of PMT Waveforms")
    ax_sum.set_xlabel("Time (us)")
    ax_sum.set_ylabel("Voltage (V)")
    ax_sum.grid(True, alpha=0.35)
    ax_sum.legend()

    if finger_plot.size == 0:
        raise RuntimeError("No finite waveforms passed the current selection cuts.")

    ax_integral.hist(finger_plot, histtype="step", bins=50)
    ax_integral.set_title("Integral of PMT Waveforms")
    ax_integral.set_xlabel("Integral of PMT Waveform (mV*ns)")
    ax_integral.set_ylabel("Count")
    #ax_integral.set_yscale("log")
    ax_integral.grid(True, alpha=0.35)

    ax_height = ax_integral.inset_axes([0.58, 0.58, 0.36, 0.34])
    ax_height.hist(max_height, histtype="step", bins=50)
    ax_height.set_title("Max Height", fontsize=9)
    ax_height.set_xlabel("V", fontsize=8)
    ax_height.set_ylabel("Count", fontsize=8)
    ax_height.tick_params(axis="both", labelsize=8)
    ax_height.grid(True, alpha=0.35)

    hist = ax_integral_height.hist2d(
        finger_plot,
        max_height,
        bins=100,
        norm=colors.LogNorm(),
        cmap="viridis",
    )
    ax_integral_height.set_title("Integral vs Maximum Height of PMT Waveforms")
    ax_integral_height.set_xlabel("Integral of PMT Waveform (mV*ns)")
    ax_integral_height.set_ylabel("Maximum Height of PMT Waveform (V)")
    fig.colorbar(hist[3], ax=ax_integral_height, label="Count")

    fit_mask = np.isfinite(finger_plot) & np.isfinite(max_height)
    if np.count_nonzero(fit_mask) >= 2:
        fit_slope, fit_intercept = np.polyfit(finger_plot[fit_mask], max_height[fit_mask], 1)
        fit_x = np.linspace(np.min(finger_plot[fit_mask]), np.max(finger_plot[fit_mask]), 200)
        fit_y = fit_slope * fit_x + fit_intercept

        fit_prediction = fit_slope * finger_plot[fit_mask] + fit_intercept
        residuals = max_height[fit_mask] - fit_prediction
        total_variance = max_height[fit_mask] - np.mean(max_height[fit_mask])
        total_variance_sum = np.sum(total_variance**2)
        r_squared = np.nan
        if total_variance_sum > 0:
            r_squared = 1 - np.sum(residuals**2) / total_variance_sum

        print("Linear fit: max_height = slope * integral + intercept")
        print(f"  slope     = {fit_slope:.6e} V/(mV*ns)")
        print(f"  intercept = {fit_intercept:.6e} V")
        print(f"  R^2       = {r_squared:.6f}")

        ax_integral_height.plot(fit_x, fit_y, color="red", linewidth=2, label="Linear fit")
        ax_integral_height.legend()

    fig.tight_layout()
    output_name = ROOT_FILE.stem.removeprefix("waveforms_")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / f"{output_name}.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
