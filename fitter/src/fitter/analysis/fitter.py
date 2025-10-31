from functools import lru_cache
import gc
from ..reactor.reactor_expected import ReactorExpected
from ..common.cache_monitor import CacheMonitor
import torch
import time
from ..config import GlobalConfig as gcfg
import numpy as np
from ..physics.surprob_fit_sys import SurProb_e2e
from ..common.my_interpolator import torch_2d_interp

torch.set_default_device(gcfg.device)
torch.set_default_dtype(torch.float64)


class Fitter:
    def __init__(
        self,
        year=1,
        MO="NO",
        rebin_factor=1,
        n_E_nu_bins=5600,  # default step 0.002 MeV (13.0-1.8)/5600≈0.002
        n_E_dep_bins=5600,  # default step 0.002 MeV (12.0-0.8)/5600≈0.002
        n_E_d_bins=560,
        n_E_p_bins=560,
    ):
        self.data_MO = MO
        self.NO_params_NO = (
            2.5303e-3,  # dmsq31 (Zhang Han) 2.5303e-3
            0.30702,  # sinsq12
            7.5303e-5,  # dmsq21
            2.189e-2,  # sinsq13
            0,  # alpha_RC
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,  # alpha_RUCs
            0,  # alpha_D
            0,
            0,
            0,
            0,
            0,
            0,  # alpha_l_pull0
            0,
            0,
            0,
            0,  # alpha_B
            0,  # alpha_Eres_a
            0,  # alpha_Eres_b
            0,  # alpha_Eres_c
            0,  # alpha_rho_ME
        )
        self.NO_params_IO = (
            -2.5051217e-3,  # dmsq31 for inverted ordering
            0.30702,  # sinsq12
            7.5303e-5,  # dmsq21
            2.189e-2,  # sinsq13
            0,  # alpha_RC
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,  # alpha_RUCs
            0,  # alpha_D
            0,
            0,
            0,
            0,
            0,
            0,
            0,  # alpha_l_pull0
            0,
            0,
            0,
            0,  # alpha_B
            0,  # alpha_Eres_a
            0,  # alpha_Eres_b
            0,  # alpha_Eres_c
            0,  # alpha_rho_ME
        )

        self.IO_params_NO = (
            0.00249578,
            0.30702,
            7.52984e-05,
            0.0217329,
            0,  # alpha_RC
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,  # alpha_RUCs
            0,  # alpha_D
            0,
            0,
            0,
            0,
            0,
            0,
            0,  # alpha_l_pull0
            0,
            0,
            0,
            0,  # alpha_B
            0,  # alpha_Eres_a
            0,  # alpha_Eres_b
            0,  # alpha_Eres_c
            0,  # alpha_rho_ME
        )
        self.IO_params_IO = (
            -2.454e-3,
            0.307,
            7.53e-05,
            0.0219,
            0,  # alpha_RC
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,  # alpha_RUCs
            0,  # alpha_D
            0,
            0,
            0,
            0,
            0,
            0,
            0,  # alpha_l_pull0
            0,
            0,
            0,
            0,  # alpha_B
            0,  # alpha_Eres_a
            0,  # alpha_Eres_b
            0,  # alpha_Eres_c
            0,  # alpha_rho_ME
        )
        self.names = (
            "dmsq31",
            "sinsq12",
            "dmsq21",
            "sinsq13",
            "alpha_RC",
            "alpha_reactor1",
            "alpha_reactor2",
            "alpha_reactor3",
            "alpha_reactor4",
            "alpha_reactor5",
            "alpha_reactor6",
            "alpha_reactor7",
            "alpha_reactor8",
            "alpha_reactor9",
            "alpha_D",
            "alpha_bkg0",
            "alpha_bkg1",
            "alpha_bkg2",
            "alpha_bkg3",
            "alpha_bkg4",
            "alpha_bkg5",
            "alpha_bkg6",
            "alpha_Eres_a",
            "alpha_Eres_b",
            "alpha_Eres_c",
            "alpha_l_pull0",
            "alpha_l_pull1",
            "alpha_l_pull2",
            "alpha_l_pull3",
        )

        self.syst_flags = {
            "SNF": True,
            "NonEq": True,
            "Eres": True,
            "l": True,
            "rho_ME": True,
            "bkg": True,
            "RUCs": True,
            "RC": True,
            "D": True,
        }
        self.year = year
        self.rea = ReactorExpected(
            day=self.year * 334.812,
            rebin_factor=rebin_factor,
            n_E_nu_bins=n_E_nu_bins,  # default step 0.002 MeV (13.0-1.8)/5600≈0.002
            n_E_dep_bins=n_E_dep_bins,  # default step 0.002 MeV (12.0-0.8)/5600≈0.002
            n_E_d_bins=n_E_d_bins,
            n_E_p_bins=n_E_p_bins,
        )
        self.n_bin = n_E_p_bins
        self.bin2bin_sig = torch.tensor(self.rea.get_bin2bin_sig())
        self.bin2bin_bkg_abs = torch.sum(
            self.rea.get_bin2bin_bkg_abs(day_bkg=self.year * 334.812 / 11 * 12), axis=0
        )
        self.sinsq12 = None
        self.sinsq13 = None
        self.dmsq21 = None
        self.dmsq32 = None
        self.dmsq31 = None
        self.y_obs = None
        self.save_bft_spec = True
        self.y_bft = None

    def clear_reactor_cache(self):
        """在每个拟合迭代后调用此方法"""
        # Explicitly release tensors
        if hasattr(self.rea, "cached_tensors"):
            for tensor in self.rea.cached_tensors:
                if isinstance(tensor, torch.Tensor):
                    del tensor
        self.rea.clear_all_cache()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        if torch.cuda.is_available():
            print(
                f"GPU Memory after clear: {torch.cuda.memory_allocated() / 1e6:.2f} MB"
            )
        print(f"Reactor cache cleared (CUDA available: {torch.cuda.is_available()})")

    def set_initial_params(self, normal=None, inverted=None):
        if normal is not None:
            self.NO_params_NO = normal
        if inverted is not None:
            self.NO_params_IO = inverted

    def set_syst_flags(self, flags_dict):
        self.syst_flags.update(flags_dict)

    def get_obs_spectrum(self, seed=0):
        if self.data_MO == "NO":
            self.sinsq12 = gcfg.sinsq12_NO
            self.sinsq13 = gcfg.sinsq13_NO
            self.dmsq21 = gcfg.dmsq21_NO
            self.dmsq32 = gcfg.dmsq32_NO
            self.dmsq31 = gcfg.dmsq31_NO
        elif self.data_MO == "IO":
            self.sinsq12 = gcfg.sinsq12_IO
            self.sinsq13 = gcfg.sinsq13_IO
            self.dmsq21 = gcfg.dmsq21_IO
            self.dmsq32 = gcfg.dmsq32_IO
            self.dmsq31 = gcfg.dmsq31_IO
        else:
            raise ValueError("Invalid value for MO. Expected 'NO' or 'IO'.")
        print("Parameters used to generate data:")
        print("dmsq31: ", self.dmsq31)
        print("dmsq21: ", self.dmsq21)
        print("sinsq12: ", self.sinsq12)
        print("sinsq13: ", self.sinsq13)
        self.rea.SetRndSeed(seed)
        alpha_Eres_a = 0
        alpha_Eres_b = 0
        alpha_Eres_c = 0
        alpha_rho_ME = 0
        alpha_bkg0 = 0
        alpha_bkg1 = 0
        alpha_bkg2 = 0
        alpha_bkg3 = 0
        alpha_bkg4 = 0
        alpha_bkg5 = 0
        alpha_bkg6 = 0
        alpha_reactor1 = 0
        alpha_reactor2 = 0
        alpha_reactor3 = 0
        alpha_reactor4 = 0
        alpha_reactor5 = 0
        alpha_reactor6 = 0
        alpha_reactor7 = 0
        alpha_reactor8 = 0
        alpha_reactor9 = 0
        alpha_RC = 0
        alpha_D = 0

        rg = np.random.default_rng(seed)
        if seed:
            if self.syst_flags.get("Eres", True):
                alpha_Eres_a = rg.normal(0, gcfg.gsigma_Eres_a)
                alpha_Eres_b = rg.normal(0, gcfg.gsigma_Eres_b)
                alpha_Eres_c = rg.normal(0, gcfg.gsigma_Eres_c)
            # if self.syst_flags.get("rho_ME", True):
            #     alpha_rho_ME = rg.normal(0, gcfg.gsigma_rho_ME)
            if self.syst_flags.get("bkg", True):
                alpha_bkg0 = rg.normal(0, gcfg.m_sigma_bkg_rate[0])
                alpha_bkg1 = rg.normal(0, gcfg.m_sigma_bkg_rate[1])
                alpha_bkg2 = rg.normal(0, gcfg.m_sigma_bkg_rate[2])
                alpha_bkg3 = rg.normal(0, gcfg.m_sigma_bkg_rate[3])
                alpha_bkg4 = rg.normal(0, gcfg.m_sigma_bkg_rate[4])
                alpha_bkg5 = rg.normal(0, gcfg.m_sigma_bkg_rate[5])
                alpha_bkg6 = rg.normal(0, gcfg.m_sigma_bkg_rate[6])
            if self.syst_flags.get("RUCs", True):
                alpha_reactor1 = rg.normal(0, gcfg.msigma_ReaUnC[0])
                alpha_reactor2 = rg.normal(0, gcfg.msigma_ReaUnC[1])
                alpha_reactor3 = rg.normal(0, gcfg.msigma_ReaUnC[2])
                alpha_reactor4 = rg.normal(0, gcfg.msigma_ReaUnC[3])
                alpha_reactor5 = rg.normal(0, gcfg.msigma_ReaUnC[4])
                alpha_reactor6 = rg.normal(0, gcfg.msigma_ReaUnC[5])
                alpha_reactor7 = rg.normal(0, gcfg.msigma_ReaUnC[6])
                alpha_reactor8 = rg.normal(0, gcfg.msigma_ReaUnC[7])
                alpha_reactor9 = rg.normal(0, gcfg.msigma_ReaUnC[8])
            if self.syst_flags.get("RC", True):
                alpha_RC = rg.normal(0, gcfg.m_sigma_ReaCorr)
            if self.syst_flags.get("D", True):
                alpha_D = rg.normal(0, gcfg.gsigma_DetectorCorr)

        self.y_obs = torch.tensor(
            self.rea.get_signal_plus_background(
                self.dmsq31,
                self.sinsq12,
                self.dmsq21,
                self.sinsq13,
                alpha_Eres_a,
                alpha_Eres_b,
                alpha_Eres_c,
                alpha_rho_ME,
                alpha_bkg0,
                alpha_bkg1,
                alpha_bkg2,
                alpha_bkg3,
                alpha_bkg4,
                alpha_bkg5,
                alpha_bkg6,
                alpha_reactor1,
                alpha_reactor2,
                alpha_reactor3,
                alpha_reactor4,
                alpha_reactor5,
                alpha_reactor6,
                alpha_reactor7,
                alpha_reactor8,
                alpha_reactor9,
                alpha_RC,
                alpha_D,
            )
        )

        self.rea.SetRndSeed(0)
        return self.y_obs

    def get_pull_sample(self, seed=0):
        rg = np.random.default_rng(seed)
        if seed:
            if gcfg.IfConstrainSinsq13:
                gcfg.sinsq13_0 = rg.normal(gcfg.sinsq13_NO, gcfg.sigma_sinsq13)
                print("sampled sinsq13_0: ", gcfg.sinsq13_0)
            if gcfg.IfConstrainDmsq31:
                gcfg.dmsq31_0_NO = rg.normal(
                    gcfg.dmsq31_NO, gcfg.grsigma_dmsq31 * gcfg.dmsq31_NO
                )
                gcfg.dmsq31_0_IO = rg.normal(
                    gcfg.dmsq31_IO, gcfg.grsigma_dmsq31 * gcfg.dmsq31_IO * -1.0
                )
                print("sampled dmsq31_0_NO: ", gcfg.dmsq31_0_NO)
                print("sampled dmsq31_0_IO: ", gcfg.dmsq31_0_IO)
        else:
            print("seed is 0, use the external measurement")
            if gcfg.IfConstrainSinsq13:
                print("sinsq13_0: ", gcfg.sinsq13_0)
            if gcfg.IfConstrainDmsq31:
                print("dmsq31_0_NO: ", gcfg.dmsq31_0_NO)
                print("dmsq31_0_IO: ", gcfg.dmsq31_0_IO)

    def _chi2_standard(self, par):
        (
            dmsq31,
            sinsq12,
            dmsq21,
            sinsq13,
            alpha_RC,
            alpha_reactor1,
            alpha_reactor2,
            alpha_reactor3,
            alpha_reactor4,
            alpha_reactor5,
            alpha_reactor6,
            alpha_reactor7,
            alpha_reactor8,
            alpha_reactor9,
            alpha_D,
            alpha_bkg0,
            alpha_bkg1,
            alpha_bkg2,
            alpha_bkg3,
            alpha_bkg4,
            alpha_bkg5,
            alpha_bkg6,
            alpha_Eres_a,
            alpha_Eres_b,
            alpha_Eres_c,
            alpha_l_pull0,
            alpha_l_pull1,
            alpha_l_pull2,
            alpha_l_pull3,
        ) = par
        if gcfg.monitoring:
            self.rea.cache_monitor.enable_monitoring(True)
            self.rea.prob.cache_monitor.enable_monitoring(True)

        chisq_rea = 0

        if gcfg.IfConstrainSinsq13:
            chisq_rea += ((sinsq13 - gcfg.sinsq13_0) / gcfg.sigma_sinsq13) ** 2
        if gcfg.IfConstrainDmsq31:
            if dmsq31 > 0:
                chisq_rea += (
                    (dmsq31 - gcfg.dmsq31_0_NO)
                    / (gcfg.dmsq31_0_NO * gcfg.grsigma_dmsq31)
                ) ** 2
            else:
                chisq_rea += (
                    (dmsq31 - gcfg.dmsq31_0_IO)
                    / (-1.0 * gcfg.dmsq31_0_IO * gcfg.grsigma_dmsq31)
                ) ** 2

        if gcfg.IfConstrainRC:
            chisq_rea += alpha_RC * alpha_RC / gcfg.m_sigma_ReaCorr**2
        if gcfg.IfConstrainRUCs:
            chisq_rea += alpha_reactor1 * alpha_reactor1 / gcfg.msigma_ReaUnC[0] ** 2
            chisq_rea += alpha_reactor2 * alpha_reactor2 / gcfg.msigma_ReaUnC[1] ** 2
            chisq_rea += alpha_reactor3 * alpha_reactor3 / gcfg.msigma_ReaUnC[2] ** 2
            chisq_rea += alpha_reactor4 * alpha_reactor4 / gcfg.msigma_ReaUnC[3] ** 2
            chisq_rea += alpha_reactor5 * alpha_reactor5 / gcfg.msigma_ReaUnC[4] ** 2
            chisq_rea += alpha_reactor6 * alpha_reactor6 / gcfg.msigma_ReaUnC[5] ** 2
            chisq_rea += alpha_reactor7 * alpha_reactor7 / gcfg.msigma_ReaUnC[6] ** 2
            chisq_rea += alpha_reactor8 * alpha_reactor8 / gcfg.msigma_ReaUnC[7] ** 2
            chisq_rea += alpha_reactor9 * alpha_reactor9 / gcfg.msigma_ReaUnC[8] ** 2
        if gcfg.IfConstrainD:
            chisq_rea += alpha_D * alpha_D / gcfg.gsigma_DetectorCorr**2
        if gcfg.IfConstrainBkg:
            chisq_rea += alpha_bkg0**2 / gcfg.m_sigma_bkg_rate[0] ** 2
            chisq_rea += alpha_bkg1**2 / gcfg.m_sigma_bkg_rate[1] ** 2
            chisq_rea += alpha_bkg2**2 / gcfg.m_sigma_bkg_rate[2] ** 2
            chisq_rea += alpha_bkg3**2 / gcfg.m_sigma_bkg_rate[3] ** 2
            chisq_rea += alpha_bkg4**2 / gcfg.m_sigma_bkg_rate[4] ** 2
            chisq_rea += alpha_bkg5**2 / gcfg.m_sigma_bkg_rate[5] ** 2
            chisq_rea += alpha_bkg6**2 / gcfg.m_sigma_bkg_rate[6] ** 2
        if gcfg.IfConstrainEres:
            chisq_rea += (alpha_Eres_a / gcfg.gsigma_Eres_a) ** 2
            chisq_rea += (alpha_Eres_b / gcfg.gsigma_Eres_b) ** 2
            chisq_rea += (alpha_Eres_c / gcfg.gsigma_Eres_c) ** 2
        # if gcfg.IfConstrainRhoME:
        #     chisq_rea += alpha_rho_ME * alpha_rho_ME / gcfg.gsigma_rho_ME**2
        Pull_2_T = 1
        Pull_2_T += alpha_RC + alpha_D
        T_nu = self.rea.get_E_p_spectrum(
            dmsq31,
            sinsq12,
            dmsq21,
            sinsq13,
            alpha_Eres_a,
            alpha_Eres_b,
            alpha_Eres_c,
            alpha_l_pull0,
            alpha_l_pull1,
            alpha_l_pull2,
            alpha_l_pull3,
            alpha_reactor1,
            alpha_reactor2,
            alpha_reactor3,
            alpha_reactor4,
            alpha_reactor5,
            alpha_reactor6,
            alpha_reactor7,
            alpha_reactor8,
            alpha_reactor9,
        )

        K_i = self.rea.get_background_spectrum(
            alpha_bkg0,
            alpha_bkg1,
            alpha_bkg2,
            alpha_bkg3,
            alpha_bkg4,
            alpha_bkg5,
            alpha_bkg6,
        )

        D_i0 = self.y_obs

        SumUnCorrErr = (
            (torch.pow(gcfg.gsigma_signal_b2b * T_nu, 2)) * 0.02 / gcfg.DataBinWidth
        )
        SumUnCorrErr += self.bin2bin_bkg_abs * 0.036 / gcfg.DataBinWidth
        SumUnCorrErr = torch.tensor(SumUnCorrErr)
        D_i = D_i0
        T_i = torch.tensor(T_nu * Pull_2_T + K_i)
        T_i = torch.where(T_i < 0, torch.ones_like(T_i) * 1e-23, T_i)
        # sqterm = torch.where(
        #     D_i == 0,
        #     2 * T_i,
        #     (D_i0 - T_i) ** 2 * (1 / D_i + 2 / (T_i + SumUnCorrErr)) / 3,
        # )
        if gcfg.test_statistic == 0:  # CNP
            SumSigmaSq_i = 3.0 / (1.0 / (D_i) + 2.0 / (T_i)) + SumUnCorrErr
            sqterm = torch.where(
                D_i == 0, 2 * T_i, (D_i - T_i) * (D_i - T_i) / SumSigmaSq_i
            )
            sqterm = sqterm[gcfg.start_bin :]
            chisq_rea += sqterm.sum()
            # logterm = torch.where(D_i == 0, torch.zeros_like(D_i), torch.log(SumSigmaSq_i))
            # chisq_rea += logterm.sum()  # + log|V|
        elif gcfg.test_statistic == 1:  # Pearson
            v = (D_i - T_i).reshape(-1, 1)
            SumSigmaSq_i = T_i + SumUnCorrErr
            cov = torch.diag(SumSigmaSq_i)
            chisq_rea += v.T @ torch.linalg.inv(cov) @ v
        elif gcfg.test_statistic == 2:  # Pearson + log|V|
            SumSigmaSq_i = T_i + SumUnCorrErr
            # cov = torch.diag(SumSigmaSq_i)
            # v = (D_i-T_i).reshape(-1, 1)
            # chisq_rea += v.T @ torch.linalg.inv(cov) @ v
            sqterm = (D_i - T_i) * (D_i - T_i) / SumSigmaSq_i
            # chisq_rea += sqterm.sum()
            # chisq_rea += torch.log(SumSigmaSq_i).sum()
            chisq_rea += (sqterm[gcfg.start_bin : gcfg.end_bin]).sum()
            chisq_rea += torch.log(SumSigmaSq_i[gcfg.start_bin : gcfg.end_bin]).sum()

        if gcfg.monitoring:
            self.rea.cache_monitor.print_stats()
            self.rea.prob.cache_monitor.print_stats()
            # self.rea.cache_monitor.reset_stats()  # safely reset statistics
        # print(f"{chisq_rea:.16e}")

        # Save best-fit expected spectrum
        if self.save_bft_spec:
            self.y_bft = T_i.numpy()
            self.y_bft_bkg_comp = self.rea.get_divided_background_spectrum(
                alpha_bkg0,
                alpha_bkg1,
                alpha_bkg2,
                alpha_bkg3,
                alpha_bkg4,
                alpha_bkg5,
                alpha_bkg6,
            )
            self.y_bft_signal = T_nu * Pull_2_T
            self.sigma2 = SumSigmaSq_i.numpy()
            self.chi2_binbybin = sqterm.numpy()

        # print('chisq_rea: ', chisq_rea)
        return chisq_rea

    def chi2(self, par):
        """统一入口方法"""
        return self._chi2_standard(par)

    def load_data_spec(self, tensor_rec_spec):
        self.y_obs = tensor_rec_spec
        print("Load real dataset, size: ", self.y_obs.shape)

    def get_bft_spec(self):
        return self.y_bft


if __name__ == "__main__":
    from iminuit import Minuit
