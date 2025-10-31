from ..config import GlobalConfig as gcfg
import torch
import uproot
from ..common.my_interpolator import torch_interp
from ..common.cache_monitor import CacheMonitor
from functools import cached_property, lru_cache, wraps

torch.set_default_device(gcfg.device)
torch.set_default_dtype(torch.float64)

"""
A high precision calibration of the nonlinear energy response at Daya Bay,
Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment,
Volume 940,
2019,
Pages 230-242,
ISSN 0168-9002,
https://doi.org/10.1016/j.nima.2019.06.031.
"""


class NonLinearity:
    def __init__(self):
        file_path = "fitter/data/data_example.root"
        with uproot.open(file_path) as file:
            # Process original_nonlinearity histogram
            hist_new = file["original_nonlinearity"]
            edges_new = hist_new.axis().edges()
            self.x_new = torch.tensor(
                (edges_new[:-1] + edges_new[1:]) / 2
            )  # Compute bin centers
            self.y_new = torch.tensor(hist_new.values())

            # Process new_nonlinearity histogram
            hist_nom = file["new_nonlinearity"]
            edges_nom = hist_nom.axis().edges()
            self.x_nom = torch.tensor((edges_nom[:-1] + edges_nom[1:]) / 2)
            self.y_nom = torch.tensor(hist_nom.values())

            # Process four pull histograms
            self.f_pull = []
            for i in range(4):
                hist_pull = file[f"nonlinearity_pull_{i + 1}"]
                self.f_pull.append(torch.tensor(hist_pull.values()))

        self.f_pull = torch.stack(self.f_pull)
        self.cache_monitor = CacheMonitor()  # New monitor instance
        self._cache_stats = {}

    @CacheMonitor.monitor
    @lru_cache(maxsize=1000)
    def __call__(
        self, E_dep, alpha_l_pull0, alpha_l_pull1, alpha_l_pull2, alpha_l_pull3
    ):
        # Move base interpolation data to the same device
        x_new = self.x_new
        y_new = self.y_new
        x_nom = self.x_nom
        y_nom = self.y_nom
        f_pull = self.f_pull

        # Perform interpolation
        f_new_interp = torch_interp(E_dep, x_new, y_new)
        f_nom_interp = torch_interp(E_dep, x_nom, y_nom)

        # Handle four pull terms
        f_pull_interp = []
        for i in range(4):
            fp = torch_interp(E_dep, x_nom, f_pull[i])
            f_pull_interp.append(fp)

        # Keep subsequent calculations as tensor operations
        f_correction = (
            alpha_l_pull0 * (f_pull_interp[0] - f_nom_interp)
            + alpha_l_pull1 * (f_pull_interp[1] - f_nom_interp)
            + alpha_l_pull2 * (f_pull_interp[2] - f_nom_interp)
            + alpha_l_pull3 * (f_pull_interp[3] - f_nom_interp)
        )

        f = f_nom_interp + f_correction
        return f * f_new_interp / f_nom_interp
