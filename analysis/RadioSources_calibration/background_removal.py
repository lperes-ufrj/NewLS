import sys
from pathlib import Path

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import trapezoid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.ReadWaveForms import load_waveforms

DATA_DIR_BKG = PROJECT_ROOT / "data" / "background"
ROOT_FILE_BKG = DATA_DIR_BKG / "waveforms_hybrid_pmt_1p1kv_cs137_labsno_ppo2g_l.root"
METADATA_FILE_BKG = DATA_DIR_BKG / "metadata_hybrid_pmt_1p1kv_cs137_labsno_ppo2g_l.json"

DATA_DIR_SOURCE = PROJECT_ROOT / "data" / "RadioSources_Calibration"
ROOT_FILE_SOURCE = DATA_DIR_SOURCE / "waveforms_hybrid_pmt_1p1kv_cs137_labsno_ppo2g_l.root"
METADATA_FILE_SOURCE = DATA_DIR_SOURCE / "metadata_hybrid_pmt_1p1kv_cs137_labsno_ppo2g_l.json"

WAVEFORM_INDICES = np.arange(20000)
PLOTS_DIR = PROJECT_ROOT / "analysis" / "plots"

BASELINE_WINDOW_MAX_US = -0.05
INTEGRATION_WINDOW_US = (-0.02, 0.03)
SIGNAL_WINDOW_US = (-0.025, 0.05)
OUT_OF_WINDOW_MAX_V = 0.15
MAX_ACCEPTED_HEIGHT_V = 0.15
MIN_ACCEPTED_HEIGHT_V = 0.0


def integrate_waveform(time_us, voltage, integration_mask):
    return trapezoid(voltage[integration_mask] * 1e3, time_us[integration_mask] * 1e3)

def baseline_mask(time_vec):
    return time_vec < BASELINE_WINDOW_MAX_US

def integration_mask(time_vec):
    return (time_vec >= INTEGRATION_WINDOW_US[0]) & (time_vec <= INTEGRATION_WINDOW_US[1])

def outside_signal_mask(time_vec):
    return (time_vec < SIGNAL_WINDOW_US[0]) | (time_vec > SIGNAL_WINDOW_US[1])

def loop_waveforms(waveforms,time):
    finger_plot = []
    max_height = []
    rejected_out_of_window = 0
    rejected_too_high = 0
    rejected_too_low = 0
    voltage_sum = []
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
            finger_plot.append(integrate_waveform(time, voltage, integration_mask))
            max_height.append(height)
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
    return voltage, voltage_sum, np.array(finger_plot), np.array(max_height)
            


def main():
    time_us_src, waveforms_src, time_window_us_src = load_waveforms(
        ROOT_FILE_SOURCE,
        WAVEFORM_INDICES,
        METADATA_FILE_SOURCE,
    )

    time_us_bkg, waveforms_bkg, time_window_us_bkg = load_waveforms(
        ROOT_FILE_BKG,
        WAVEFORM_INDICES,
        METADATA_FILE_BKG,
    )

    fig, axs = plt.subplots(3, 2, figsize=(14, 9))
    fig.suptitle(f"Analysis_wBkg_removal_{ROOT_FILE_SOURCE.stem}", fontsize=10)
    
    ax_waveforms_src = axs[0, 0]
    ax_waveforms_bkg = axs[0,1]

    ax_integral_both = axs[1,0]
    sum_waveforms_both = axs[1,1]

    ax_integral_rm_bkg = axs[2,0]
    ax_integral_fit = axs[2,1]

    voltage_src, voltage_sum_src, finger_plot_src, max_height_src = loop_waveforms(waveforms_src, time_us_src)
    voltage_bkg, voltage_sum_bkg, finger_plot_bkg, max_height_bkg = loop_waveforms(waveforms_bkg, time_us_bkg)

    ax_waveforms_src.plot(time_us_src, voltage_src, linewidth=1)
    ax_waveforms_bkg.plot(time_us_bkg, voltage_bkg, linewidth=1)

    finite_mask = np.isfinite(finger_plot) & np.isfinite(max_height)
    finger_plot = finger_plot[finite_mask]
    max_height = max_height[finite_mask]

    ax_waveforms_src.set_title("MDO2024 Waveform with Source")
    ax_waveforms_src.set_ylabel("Voltage (V)")
    if time_window_us_src is None:
        ax_waveforms_src.set_xlabel("Sample")
    else:
        ax_waveforms_src.set_xlim([-0.2, 0.2])
        ax_waveforms_src.set_xlabel("Time (us)")
    ax_waveforms_src.axvspan(
        INTEGRATION_WINDOW_US[0],
        INTEGRATION_WINDOW_US[1],
        color="tab:orange",
        alpha=0.2,
        label="Integration window",
    )
    ax_waveforms_src.grid(True, alpha=0.35)
    ax_waveforms_src.legend(title=f"Number of Waveforms: {finger_plot.size:.0f}", fontsize=8)

    ax_waveforms_bkg.set_title("MDO2024 Waveform Background")
    ax_waveforms_bkg.set_ylabel("Voltage (V)")
    if time_window_us_bkg is None:
        ax_waveforms_bkg.set_xlabel("Sample")
    else:
        ax_waveforms_bkg.set_xlim([-0.2, 0.2])
        ax_waveforms_bkg.set_xlabel("Time (us)")
    ax_waveforms_bkg.axvspan(
        INTEGRATION_WINDOW_US[0],
        INTEGRATION_WINDOW_US[1],
        color="tab:orange",
        alpha=0.2,
        label="Integration window",
    )
    ax_waveforms_bkg.grid(True, alpha=0.35)
    ax_waveforms_bkg.legend(title=f"Number of Waveforms: {finger_plot.size:.0f}", fontsize=8)


    sum_waveforms_both.plot(time_us_src, voltage_sum_src, label="Source+Background")
    sum_waveforms_both.fill_between(
        time_us_src[integration_mask],
        voltage_sum_src[integration_mask],
        alpha=0.25,
        color="tab:orange",
        label="Integrated area",
    )
    
    sum_waveforms_both.plot(time_us_src, voltage_sum_bkg, label="Background")
    sum_waveforms_both.fill_between(
        time_us_src[integration_mask],
        voltage_sum_bkg[integration_mask],
        alpha=0.25,
        color="tab:orange",
        label="Integrated area",
    )

    sum_waveforms_both.set_title("Sum of PMT Waveforms")
    sum_waveforms_both.set_xlabel("Time (us)")
    sum_waveforms_both.set_ylabel("Voltage (V)")
    sum_waveforms_both.grid(True, alpha=0.35)
    sum_waveforms_both.legend()

    if finger_plot.size == 0:
        raise RuntimeError("No finite waveforms passed the current selection cuts.")

    ax_integral_both.hist(finger_plot, histtype="step", bins=50)
    ax_integral_both.set_title("Integral of PMT Waveforms")
    ax_integral_both.set_xlabel("Integral of PMT Waveform (mV*ns)")
    ax_integral_both.set_ylabel("Count")
    #ax_integral.set_yscale("log")
    ax_integral_both.grid(True, alpha=0.35)

    ax_height_both = ax_integral_both.inset_axes([0.58, 0.58, 0.36, 0.34])
    ax_height_both.hist(max_height_src, histtype="step", bins=50)
    ax_height_both.hist()
    ax_height_both.set_title("Max Height", fontsize=9)
    ax_height_both.set_xlabel("V", fontsize=8)
    ax_height_both.set_ylabel("Count", fontsize=8)
    ax_height_both.tick_params(axis="both", labelsize=8)
    ax_height_both.grid(True, alpha=0.35)


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

