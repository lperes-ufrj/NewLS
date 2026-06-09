import configparser
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.ReadWaveForms import load_waveforms


DATA_DIR = PROJECT_ROOT / "data" / "background"
ROOT_FILE = DATA_DIR / "waveforms_hybrid_pmt_1p1kv_10k_hires_ba133_0602_background_run1.root"
METADATA_FILE = DATA_DIR / "metadata_hybrid_pmt_1p1kv_10k_hires_ba133_0602_background_run1.json"
CALIBRATION_FILE = (
    PROJECT_ROOT
    / "analysis"
    / "pmt_calibration"
    / "waveforms_hybrid_pmt_1p1kv_10k_hires_calibration_0603_aftern2_calibration.ini"
)
WAVEFORM_INDICES = np.arange(10000)
PLOTS_DIR = PROJECT_ROOT / "analysis" / "plots"

BASELINE_WINDOW_MAX_US = -0.05
INTEGRATION_WINDOW_US = (-0.02, 0.03)
SIGNAL_WINDOW_US = (-0.025, 0.05)
OUT_OF_WINDOW_MAX_V = 0.075
MAX_ACCEPTED_HEIGHT_V = 0.075
MIN_ACCEPTED_HEIGHT_V = 0.0

BINS = 90
N_PEAKS = 3
FIT_MIN = None
FIT_MAX = None
SHOW_PLOT = True


def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def exp_background(x, constant, amplitude, slope):
    return constant + amplitude * np.exp(-slope * x)


def multi_peak_model(x, constant, background_amplitude, background_slope, *peak_params):
    y = exp_background(x, constant, background_amplitude, background_slope)
    for amplitude, mean, sigma in np.reshape(peak_params, (-1, 3)):
        y = y + gaussian(x, amplitude, mean, sigma)
    return y


def integrate_waveform(time_us, voltage, integration_mask):
    return np.trapz(voltage[integration_mask] * 1e3, time_us[integration_mask] * 1e3)


def load_q_spe(calibration_file):
    config = configparser.ConfigParser()
    read_files = config.read(calibration_file)
    if not read_files:
        raise FileNotFoundError(f"Could not read calibration file: {calibration_file}")

    calibration = config["CALIBRATION"]
    for key in ("q_spe_mVns", "q_spe_mvns"):
        if key in calibration:
            return float(calibration[key])
    raise KeyError(f"No q_spe_mVns entry found in {calibration_file}")


def moving_average(values, window):
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def initial_peak_guesses(bin_centers, counts, n_peaks):
    smoothed = moving_average(counts, window=5)
    prominence = max(np.max(smoothed) * 0.035, 1.0)
    distance = max(2, len(bin_centers) // 12)
    peak_indices, properties = find_peaks(
        smoothed,
        prominence=prominence,
        distance=distance,
    )

    if peak_indices.size:
        order = np.argsort(properties["prominences"])[::-1]
        peak_indices = peak_indices[order[:n_peaks]]

    if peak_indices.size < n_peaks:
        quantiles = np.linspace(0.18, 0.82, n_peaks)
        fallback_means = np.quantile(bin_centers, quantiles)
        current_means = list(bin_centers[peak_indices])
        for mean in fallback_means:
            if len(current_means) >= n_peaks:
                break
            if not current_means or np.min(np.abs(np.array(current_means) - mean)) > 0:
                current_means.append(mean)
        peak_means = np.array(current_means[:n_peaks])
    else:
        peak_means = np.array(bin_centers[peak_indices[:n_peaks]])

    peak_means = np.sort(peak_means)
    bin_width = np.median(np.diff(bin_centers))
    default_sigma = max(3.0 * bin_width, 0.06 * (bin_centers.max() - bin_centers.min()))

    guesses = []
    for mean in peak_means:
        nearest = np.argmin(np.abs(bin_centers - mean))
        amplitude = max(counts[nearest] - np.percentile(counts, 10), 1.0)
        guesses.extend([amplitude, mean, default_sigma])
    return guesses


def fit_multi_peak_histogram(data, bins, n_peaks, fit_min=None, fit_max=None):
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    if fit_min is None:
        fit_min = max(0.0, np.percentile(data, 0.5))
    if fit_max is None:
        fit_max = np.percentile(data, 99.5)

    fit_mask = (bin_centers >= fit_min) & (bin_centers <= fit_max)
    x_fit = bin_centers[fit_mask]
    y_fit = counts[fit_mask]
    if x_fit.size < 3 * n_peaks + 3:
        raise RuntimeError("Fit range has too few populated bins for the requested peak count.")

    peak_guesses = initial_peak_guesses(x_fit, y_fit, n_peaks)
    p0 = [
        max(np.percentile(y_fit, 5), 0.0),
        max(y_fit[0] - np.percentile(y_fit, 5), 1.0),
        1.0 / max(x_fit.max() - x_fit.min(), 1.0),
        *peak_guesses,
    ]

    lower = [0.0, 0.0, 0.0]
    upper = [np.inf, np.inf, np.inf]
    min_sigma = max(0.5 * bin_width, 1e-6)
    max_sigma = max(0.45 * (fit_max - fit_min), min_sigma * 2.0)
    for _ in range(n_peaks):
        lower.extend([0.0, fit_min, min_sigma])
        upper.extend([np.inf, fit_max, max_sigma])

    popt, pcov = curve_fit(
        multi_peak_model,
        x_fit,
        y_fit,
        p0=p0,
        bounds=(lower, upper),
        sigma=np.sqrt(np.maximum(y_fit, 1.0)),
        absolute_sigma=True,
        maxfev=100000,
    )
    return popt, pcov, counts, bin_edges, fit_min, fit_max


def main():
    q_spe = load_q_spe(CALIBRATION_FILE)
    print(f"Using single photoelectron charge: {q_spe:.4g} mV*ns")
    print(f"Calibration file: {CALIBRATION_FILE}")

    time_us, waveforms, time_window_us = load_waveforms(
        ROOT_FILE,
        WAVEFORM_INDICES,
        METADATA_FILE,
    )

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f"{ROOT_FILE.stem}", fontsize=10)
    ax_waveforms, ax_integral = axs

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
        raise RuntimeError("Baseline window is empty for this waveform time axis.")
    if not np.any(integration_mask):
        raise RuntimeError("Integration window is empty for this waveform time axis.")

    finger_plot = []
    max_height = []
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

        finger_plot.append(integrate_waveform(time_us, voltage, integration_mask))
        max_height.append(height)
        ax_waveforms.plot(time_us, voltage, linewidth=0.7, alpha=0.35)

    finger_plot = np.asarray(finger_plot)
    max_height = np.asarray(max_height)
    finite_mask = np.isfinite(finger_plot) & np.isfinite(max_height)
    finger_plot = finger_plot[finite_mask]
    max_height = max_height[finite_mask]
    if finger_plot.size == 0:
        raise RuntimeError("No finite waveforms passed the current selection cuts.")

    print(
        f"Accepted {finger_plot.size} waveforms; rejected "
        f"{rejected_out_of_window} out-of-window, "
        f"{rejected_too_high} above {MAX_ACCEPTED_HEIGHT_V:g} V, and "
        f"{rejected_too_low} below {MIN_ACCEPTED_HEIGHT_V:g} V."
    )

    ax_waveforms.set_title("Selected PMT Waveforms")
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
    ax_waveforms.legend(title=f"Number of waveforms: {finger_plot.size}", fontsize=8)

    fit_result = None
    try:
        fit_result = fit_multi_peak_histogram(
            finger_plot,
            bins=BINS,
            n_peaks=N_PEAKS,
            fit_min=FIT_MIN,
            fit_max=FIT_MAX,
        )
    except RuntimeError as exc:
        print(f"Multi-peak fit skipped: {exc}")

    if fit_result is None:
        ax_integral.hist(finger_plot, bins=BINS, histtype="step", label="Data")
    else:
        popt, pcov, counts, bin_edges, fit_min, fit_max = fit_result
        ax_integral.hist(finger_plot, bins=bin_edges, histtype="step", label="Data")
        x_plot = np.linspace(fit_min, fit_max, 1200)
        ax_integral.plot(
            x_plot,
            multi_peak_model(x_plot, *popt),
            color="red",
            linewidth=2,
            label=f"Background + {N_PEAKS} peaks",
        )
        ax_integral.plot(
            x_plot,
            exp_background(x_plot, *popt[:3]),
            color="gray",
            linestyle="--",
            label="Background",
        )

        print("Peak fit results:")
        for peak_index, (amplitude, mean, sigma) in enumerate(np.reshape(popt[3:], (-1, 3)), start=1):
            ax_integral.plot(
                x_plot,
                gaussian(x_plot, amplitude, mean, sigma),
                linestyle="--",
                linewidth=1.4,
                label=f"Peak {peak_index}",
            )
            ax_integral.axvline(mean, linestyle=":", color="black", alpha=0.45)
            print(
                f"  peak {peak_index}: mean={mean:.2f} mV*ns "
                f"({mean / q_spe:.1f} PE), sigma={sigma:.2f} mV*ns, "
                f"amplitude={amplitude:.1f}"
            )

    ax_integral.set_title("Integral of PMT Waveforms")
    ax_integral.set_xlabel("Integral of PMT Waveform (mV*ns)")
    ax_integral.set_ylabel("Count")
    ax_integral.grid(True, alpha=0.35)
    ax_integral.legend(fontsize=8)

    ax_height = ax_integral.inset_axes([0.62, 0.38, 0.33, 0.30])
    ax_height.hist(max_height, histtype="step", bins=50)
    ax_height.set_title("Max Height", fontsize=9)
    ax_height.set_xlabel("V", fontsize=8)
    ax_height.set_ylabel("Count", fontsize=8)
    ax_height.tick_params(axis="both", labelsize=8)
    ax_height.grid(True, alpha=0.35)

    ax_photoelectrons = ax_integral.twiny()
    ax_photoelectrons.set_xlabel("Number of Photoelectrons", color="red")
    ax_photoelectrons.tick_params(axis="x", colors="red")
    charge_min, charge_max = ax_integral.get_xlim()
    ax_photoelectrons.set_xlim(charge_min / q_spe, charge_max / q_spe)

    fig.tight_layout()
    output_name = ROOT_FILE.stem.removeprefix("waveforms_")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / f"{output_name}_multi_peak_fit.png"
    fig.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    if SHOW_PLOT:
        plt.show()


if __name__ == "__main__":
    main()
