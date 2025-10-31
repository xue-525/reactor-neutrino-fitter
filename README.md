# Reactor Neutrino Fitter

A Python-based analysis framework for reactor neutrino experiments, specifically designed for Inverse Beta Decay (IBD) cross-section calculations and neutrino oscillation parameter fitting.

## Project Overview

This project provides tools for:
- Computing IBD differential cross-section matrices
- Calculating reactor neutrino expected spectra
- Performing neutrino oscillation parameter fitting
- Analyzing detector responses and energy reconstruction

## Project Structure

```
reactor-neutrino-fitter/
├── fitter/
│   ├── src/fitter/
│   │   ├── analysis/          # Analysis modules
│   │   │   ├── fitter.py      # Main fitting class
│   │   │   └── IBDXsec_Vissani.py  # IBD cross-section matrix generator
│   │   ├── physics/            # Physics calculations
│   │   │   ├── ibdxsec_Vissani.py      # IBD cross-section physics
│   │   │   └── surprob_fit_sys.py      # Survival probability calculations
│   │   ├── reactor/            # Reactor neutrino calculations
│   │   │   └── reactor_expected.py     # Expected spectrum calculator
│   │   ├── config/             # Configuration files
│   │   │   ├── GlobalConfig.py         # Global configuration parameters
│   │   │   └── ReactorConfig.py        # Reactor-specific settings
│   │   └── common/             # Common utilities
│   │       ├── cache_monitor.py        # Caching utilities
│   │       ├── my_interpolator.py      # Interpolation functions
│   │       └── nonlinearity_fit.py     # Nonlinearity corrections
│   ├── data/                   # Data and output directory
│   │   └── outputs/            # Generated cross-section matrices (large files, not in git)
│   ├── lib/                    # Library files (e.g., matplotlib styles)
│   └── paper_figure.ipynb      # Example analysis notebook
└── README.md
```

## Prerequisites

- Python 3.x
- PyTorch
- NumPy
- SciPy
- Matplotlib
- iminuit (for parameter fitting)
- uproot (for ROOT file handling)
- numba (for performance optimization)
- tqdm (for progress bars)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd reactor-neutrino-fitter
```

2. Install required dependencies:
```bash
pip install torch numpy scipy matplotlib iminuit uproot numba tqdm
```

## Usage

### Step 1: Generate IBD Cross-Section Matrix

**Important**: Before running any analysis, you must first generate the IBD cross-section matrix. This file is too large (>200 MB) to be stored in GitHub, so it needs to be generated locally.

Generate the cross-section matrix by running:

```bash
python3 fitter/src/fitter/analysis/IBDXsec_Vissani.py
```

This script will:
- Calculate the IBD differential cross-section matrix for neutrino energies from 1.8 to 13.0 MeV
- Generate a matrix with configurable binning (default: 5600 bins, corresponding to ~0.002 MeV step size)
- Save the output to `fitter/data/outputs/IBDXsec_matrix_enu5600_test.pt`

**Note**: This computation can be memory-intensive and time-consuming. The default configuration with 5600 bins produces a ~240 MB output file.

You can modify the binning configuration in the script by changing the `bin_list` parameter at the bottom of the file.

### Step 2: Configure Analysis Parameters

Edit `fitter/src/fitter/config/GlobalConfig.py` to configure:
- Data file paths
- Energy binning parameters
- Detector parameters
- Oscillation parameters
- Systematic uncertainties

Key parameters:
- `data_path`: Path to the generated IBD cross-section matrix
- `file_path`: Path to input ROOT data files
- `n_E_nu_bins`: Number of neutrino energy bins (default: 5600)
- `n_E_dep_bins`: Number of deposited energy bins (default: 5600)

### Step 3: Run Analysis

#### Example 1: Calculate Expected Spectrum

```python
from fitter.src.fitter.reactor.reactor_expected import ReactorExpected
import matplotlib.pyplot as plt

# Initialize reactor expected spectrum calculator
reactor = ReactorExpected(
    n_E_nu_bins=5600,   # Neutrino energy bins (step ~0.002 MeV)
    n_E_dep_bins=5600,  # Deposited energy bins (step ~0.002 MeV)
    n_E_d_bins=560,     # Detector response bins
    n_E_p_bins=560,     # Reconstructed energy bins
)

# Get reconstructed energy spectrum
E_p_spectrum = reactor.get_E_p_spectrum()

# Plot results
import matplotlib.pyplot as plt
plt.plot(reactor.E_p, E_p_spectrum)
plt.xlabel("Energy (MeV)")
plt.ylabel("Counts (/ 20keV / year)")
plt.title("Reconstructed Energy Spectrum")
plt.show()
```

#### Example 2: Perform Parameter Fitting

```python
from iminuit import Minuit
from fitter.src.fitter.analysis.fitter import Fitter
from fitter.src.fitter.config import GlobalConfig as gcfg

# Initialize fitter
fitter = Fitter(year=1, MO="NO")

# Generate or load observed spectrum
fitter.get_obs_spectrum(gseed=1)  # gseed=0 for Asimov, >0 for toy data

# Set up Minuit for minimization
m = Minuit(fitter.chi2, fitter.NO_params_NO, name=fitter.names)
m.tol = 0.01

# Perform fit
results = m.migrad()
print(results)
```

## Key Components

### IBDXsec_Vissani.py
Generates the IBD differential cross-section matrix. This is a computationally intensive step that must be run before any analysis. The output matrix maps neutrino energy to deposited energy probabilities.

### ReactorExpected
Calculates the expected reactor neutrino spectrum, including:
- Neutrino energy spectrum from reactors
- IBD interaction rates
- Detector response and energy reconstruction
- Background contributions

### Fitter
Main analysis class for parameter fitting:
- Chi-squared calculation with systematic uncertainties
- Support for neutrino oscillation parameters (θ₁₂, θ₁₃, Δm²₂₁, Δm²₃₁)
- Reactor and detector systematic uncertainties
- Background rate and shape uncertainties

## Configuration

Key configuration files:
- **GlobalConfig.py**: Global analysis parameters, systematic uncertainties, test statistics
- **ReactorConfig.py**: Reactor-specific parameters (power, distance, etc.)

## Output Files

Generated files are stored in `fitter/data/outputs/`:
- IBD cross-section matrices (`.pt` format, PyTorch tensors)
- These files are excluded from git due to their large size (>100 MB)

## Notes

- The default energy binning (5600 bins) provides ~0.002 MeV resolution
- Memory requirements depend on the number of bins used
- The framework supports both Normal Ordering (NO) and Inverted Ordering (IO)
- Systematic uncertainties include reactor correlations, detector effects, and background modeling

## Citation

If you use this code in your research, please cite the relevant papers for the physical models and analysis methods.

## License

[Specify license here]

