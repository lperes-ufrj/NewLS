import sys
from pathlib import Path
import configparser
from datetime import datetime

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.ReadWaveForms import load_waveforms

DATA_DIR = PROJECT_ROOT / "data" / "PMT_Calibration"
ROOT_FILE = DATA_DIR / "waveforms_hybrid_pmt_1p1kv_calibration_labdet_ppo2g_l.root"
METADATA_FILE = DATA_DIR / "metadata_hybrid_pmt_1p1kv_calibration_labdet_ppo2g_l.json"
WAVEFORM_INDICES = np.arange(15000)
PLOTS_DIR = PROJECT_ROOT / "analysis" / "plots"

BASELINE_WINDOW_MAX_US = -0.05
INTEGRATION_WINDOW_US = (-0.02, 0.05)
SIGNAL_WINDOW_US = (-0.025, 0.05)
OUT_OF_WINDOW_MAX_V = 0.001
MAX_ACCEPTED_HEIGHT_V = 0.0062
MIN_ACCEPTED_HEIGHT_V = 0.0005

def gaussian(x, A, mu, sigma):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def two_gaussian(x, A0, mu0, sigma0, A1, mu1, sigma1):
    g0 = A0 * np.exp(-(x - mu0)**2 / (2 * sigma0**2))
    g1 = A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
    return g0 + g1

def integrate_waveform(time_us, voltage, integration_mask):
    return trapezoid(voltage[integration_mask] * 1e3, time_us[integration_mask] * 1e3)

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

    finger_plot = []
    max_height = []
    voltage_sum = np.zeros_like(time_us)
    rejected_out_of_window = 0

    for waveform_index, voltage in waveforms.items():
        baseline = np.mean(voltage[baseline_mask])
        voltage = -(voltage - baseline)
        if np.any(np.abs(voltage[outside_signal_mask]) > OUT_OF_WINDOW_MAX_V):
            rejected_out_of_window += 1
            continue
        height = np.max(voltage)

        if height > MAX_ACCEPTED_HEIGHT_V:
            continue
        if height < MIN_ACCEPTED_HEIGHT_V:
            continue

        voltage_sum += voltage

        #print(f"div: {voltage[5]-voltage[4]} V at {time_us[5]:.3f} us")
        finger_plot.append(integrate_waveform(time_us, voltage, integration_mask))
        max_height.append(height)

        ax_waveforms.plot(time_us, voltage, linewidth=1)

    finger_plot = np.array(finger_plot)
    max_height = np.array(max_height)
    print(
        f"Rejected {rejected_out_of_window} waveforms with voltage "
        f"> {OUT_OF_WINDOW_MAX_V:g} V outside "
        f"{SIGNAL_WINDOW_US[0]:g} to {SIGNAL_WINDOW_US[1]:g} us."
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
###############################################################################################

    # ----------------------------------------------------
    # 1. Histogram data
    # ----------------------------------------------------
    bins = 50

    counts, bin_edges = np.histogram(finger_plot, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    # ----------------------------------------------------
    # 2. Restrict fit range
    #    Adjust these numbers to your plot
    # ----------------------------------------------------
    fit_min = -15
    fit_max = 80

    fit_mask = (bin_centers > fit_min) & (bin_centers < fit_max)

    x_fit_data = bin_centers[fit_mask]
    y_fit_data = counts[fit_mask]

    # ----------------------------------------------------
    # 3. Initial guesses
    #    From your plot:
    #    pedestal is around ~ -3 to 0 mV ns
    #    SPE peak is around ~ 12 to 15 mV ns
    # ----------------------------------------------------
    p0 = [
        np.max(y_fit_data), -3.0, 3.0,      # pedestal: A0, mu0, sigma0
        np.max(y_fit_data) / 2, 15.0, 4.0   # SPE: A1, mu1, sigma1
    ]

    # Optional bounds to make the fit more stable
    bounds = (
        [0, -15, 0.1, 0,   2, 0.1],   # lower bounds
        [np.inf, 5, 20, np.inf, 80, 20]  # upper bounds
    )

    # ----------------------------------------------------
    # 4. Fit
    # ----------------------------------------------------
    popt, pcov = curve_fit(
        two_gaussian,
        x_fit_data,
        y_fit_data, 
        p0=p0, bounds=bounds, maxfev=10000
    )

    A0, mu0, sigma0, A1, mu1, sigma1 = popt

    Q_SPE = mu1 - mu0

    print("Pedestal mean =", mu0, "mV ns")
    print("SPE mean      =", mu1, "mV ns")
    print("Q_SPE         =", Q_SPE, "mV ns")
    print("Pedestal sigma =", sigma0, "mV ns")
    print("SPE sigma      =", sigma1, "mV ns")

    # Write calibration parameters to INI file
    config = configparser.ConfigParser()
    config['FILES'] = {
        'root_file': str(ROOT_FILE),
        'metadata_file': str(METADATA_FILE),
        'sample_name': ROOT_FILE.stem,
    }
    config['CALIBRATION'] = {
        'pedestal_mean_mVns': f"{mu0}",
        'spe_mean_mVns': f"{mu1}",
        'q_spe_mVns': f"{Q_SPE}",
        'pedestal_sigma_mVns': f"{sigma0}",
        'spe_sigma_mVns': f"{sigma1}",
        'generated': datetime.utcnow().isoformat() + 'Z'
    }

    #PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    ini_path = f"{ROOT_FILE.stem}_calibration.ini"
    with open(ini_path, 'w') as cfgfile:
        config.write(cfgfile)
    print(f"Wrote calibration to {ini_path}")

    ax_integral.hist(finger_plot, histtype="step", bins=bins, label="Data")
    x_plot = np.linspace(fit_min, fit_max, 1000)

    ax_integral.plot(x_plot, two_gaussian(x_plot, *popt), label="Pedestal + SPE fit")
    ax_integral.plot(x_plot, gaussian(x_plot, A0, mu0, sigma0), "--", label="Pedestal")
    ax_integral.plot(x_plot, gaussian(x_plot, A1, mu1, sigma1), "--", label="SPE")

    ax_integral.axvline(mu0, linestyle=":", label=r"$\mu_0$")
    ax_integral.axvline(mu1, linestyle=":", label=r"$\mu_1$")

    ax_integral.set_title("Integral of PMT Waveforms")
    ax_integral.set_xlabel("Integral of PMT Waveform (mV*ns)")
    ax_integral.set_ylabel("Count")
    # ax_integral.set_yscale("log")
    ax_integral.grid(True, alpha=0.35)
    ax_integral.legend()

##############################################################################################
    ax_height = ax_integral_height
    ax_height.hist(max_height, histtype="step", bins=50)
    ax_height.set_title("Max Height")
    ax_height.set_xlabel("Maximum Height of PMT Waveform (V)")
    ax_height.set_ylabel("Count")
    ax_height.grid(True, alpha=0.35)

    ax_height_integral = ax_height.inset_axes([0.50, 0.50, 0.45, 0.42])
    hist = ax_height_integral.hist2d(
        finger_plot,
        max_height,
        bins=100,
        norm=colors.LogNorm(),
        cmap="viridis",
    )
    ax_height_integral.set_title("Integral vs Height", fontsize=9)
    ax_height_integral.set_xlabel("mV*ns", fontsize=8)
    ax_height_integral.set_ylabel("V", fontsize=8)
    ax_height_integral.tick_params(axis="both", labelsize=8)

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

        ax_height_integral.plot(fit_x, fit_y, color="red", linewidth=1.5, label="Linear fit")
        ax_height_integral.legend(fontsize=7)

    fig.tight_layout()
    output_name = ROOT_FILE.stem.removeprefix("waveforms_")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / f"{output_name}.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
