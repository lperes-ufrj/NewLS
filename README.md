# NewLS

Python tools for acquiring, reading, plotting, and analyzing waveforms to PMT calibration and characterize scintillation signals.

## Project Layout

```text
ScopeInterface/
  scope_mdo3034_acquisition.py   Acquire waveforms from the oscilloscope
  plot_waveform.py               Quick plotting script using the reader library

src/
  ReadWaveForms.py               Reusable functions for reading ROOT waveform files

analysis/
  pmt_calibration/
    pmt_calibration.py           PMT calibration plotting/analysis script

data/
  PMT_Calibration/               ROOT, CSV, and metadata files
```

## Requirements

Core analysis and plotting:

```bash
pip install numpy matplotlib uproot
```

Oscilloscope acquisition also needs PyVISA and a working VISA backend:

```bash
pip install pyvisa
```

## Reading Waveforms

Use `src/ReadWaveForms.py` as the library layer. The main helper is `load_waveforms`, which returns:

- `time_us`: time axis in microseconds
- `waveforms`: dictionary mapping waveform index to voltage array
- `time_window_us`: display window from metadata, or `None`

Example:

```python
from pathlib import Path

from src.ReadWaveForms import load_waveforms

data_dir = Path("data/PMT_Calibration")
root_file = data_dir / "waveforms_hybrid_pmt_calibration_1p2kv.root"
metadata_file = data_dir / "metadata_hybrid_pmt_calibration_1p2kv.json"

time_us, waveforms, time_window_us = load_waveforms(
    root_file,
    [18, 55, 360],
    metadata_file,
)
```

## Plotting

Run the quick waveform plotter from the project root:

```bash
python ScopeInterface/plot_waveform.py
```

Run the PMT calibration analysis script:

```bash
python analysis/pmt_calibration/pmt_calibration.py
```

Both scripts import the reader functions from `src/ReadWaveForms.py`.

## Acquiring New Data

The acquisition script is configured for a Tektronix MDO3034:

```bash
python ScopeInterface/scope_mdo3034_acquisition.py
```

Before running it, check these constants inside `scope_mdo3034_acquisition.py`:

- `SAMPLE_LABEL`
- `VISA_RESOURCE`
- `WAVEFORMS_TO_READ`

The script saves:

- a CSV preview of the first waveform
- a ROOT file containing all waveform branches
- a JSON metadata file with scale and acquisition settings

## Data Format

ROOT files contain a `waveforms` tree with branches named like:

```text
waveform_00000
waveform_00001
...
```

Metadata JSON files include the display time window, oscilloscope scale settings, number of waveforms, and points per waveform.

## Notes

- Run scripts from the project root when possible.
- `src/ReadWaveForms.py` should stay free of plotting code so analysis scripts can reuse it cleanly.
- Large generated data files should be kept under `data/PMT_Calibration/`.
