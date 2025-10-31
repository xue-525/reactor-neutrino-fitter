# This py code is for crosschecking the different formula of osc

from functools import lru_cache
from ..common.cache_monitor import CacheMonitor
import torch
from ..config import GlobalConfig as gcfg

torch.set_default_device(gcfg.device)
torch.set_default_dtype(torch.float64)


class SurProb_e2e:
    """Calculate the Survival Probability of reactor neutrino (anti-nu_e 2 anti-nu_e)"""

    def __init__(self):
        """Constructor"""
        self._rho_rock = 2.45e3  # 2.45e3 kg/m^3
        # self._rho_rock = 2.64e3 # 2.45e3 kg/m^3
        self._m_unified_atomic_kg = (
            1.6605390666e-27  # unified atomic mass unit= 1.6605390666e-27 kg
        )
        self._GF = 1.1663787e-5  # GeV^-2
        self._hbar_C = 197.3269804
        self.cache_monitor = CacheMonitor()  # New monitor instance
        self._cache_stats = {}

    @CacheMonitor.monitor
    @lru_cache(maxsize=1000)
    def surprob_vacuum(
        self, neutrino_energies, baseline, dmsq31, sinsq12, dmsq21, sinsq13
    ):
        Delta21 = (
            1.266932679815373 * dmsq21 * baseline / neutrino_energies
        )  # eV^2 ,MeV ,m
        Delta31 = 1.266932679815373 * dmsq31 * baseline / neutrino_energies
        Delta32 = (
            1.266932679815373 * (dmsq31 - dmsq21) * baseline / neutrino_energies
        )  # eV^2 ,MeV ,m
        prob = (
            1.0
            - 4
            * sinsq13
            * (1 - sinsq13)
            * (
                (1 - sinsq12) * torch.pow(torch.sin(Delta31), 2.0)
                + sinsq12 * torch.pow(torch.sin(Delta32), 2.0)
            )
            - (1 - sinsq13) ** 2
            * 4
            * sinsq12
            * (1 - sinsq12)
            * torch.pow(torch.sin(Delta21), 2.0)
        )

        return prob

    def clear_cache(self):
        """自动清理所有缓存方法"""
        for attr in dir(self):
            try:
                method = getattr(self, attr)
                # Access the original decorated method via __wrapped__
                wrapped_method = getattr(method, "__wrapped__", method)
                if callable(wrapped_method) and hasattr(wrapped_method, "cache_clear"):
                    wrapped_method.cache_clear()
            except Exception as e:
                print(f"Error clearing cache for {attr}: {str(e)}")
