from scipy.interpolate import interp1d
from functools import cached_property, lru_cache, wraps
from ..physics.surprob_fit_sys import SurProb_e2e
import time
from ..common.nonlinearity_fit import NonLinearity
from ..common.my_interpolator import torch_interp
from ..common.cache_monitor import CacheMonitor
import torch
import uproot
from ..config import GlobalConfig as gcfg
from ..config import ReactorConfig as rcfg
import sys
import weakref
from scipy.linalg import blas
import numpy as np
from scipy.special import erf

torch.set_default_device(gcfg.device)
torch.set_default_dtype(torch.float64)


class ReactorExpected:
    def __init__(
        self,
        day=334.812,
        rebin_factor=1,
        n_E_nu_bins=5600,  # default step 0.002 MeV (13.0-1.8)/5600≈0.002
        n_E_dep_bins=5600,  # default step 0.002 MeV (12.0-0.8)/5600≈0.002
        n_E_d_bins=560,
        n_E_p_bins=560,
    ):
        self.file = uproot.open(gcfg.file_path)

        # Parameterized binning logic
        def calc_bins(start, end, n_bins):
            step = (end - start) / n_bins
            return torch.arange(start, end + step * 0.1, step)  # + step*0.1 ensures the endpoint is included

        # Neutrino energy binning
        self.E_nu_edges = calc_bins(1.8, 13.0, n_E_nu_bins)
        self.E_nu = (self.E_nu_edges[:-1] + self.E_nu_edges[1:]) / 2

        # Deposited energy binning
        self.E_dep_edges = calc_bins(0.8, 12.0, n_E_dep_bins)
        self.E_dep = (self.E_dep_edges[:-1] + self.E_dep_edges[1:]) / 2

        # Detector response matrix energy binning
        self.E_d_edges = calc_bins(0.8, 12.0, n_E_d_bins)
        self.E_d = (self.E_d_edges[:-1] + self.E_d_edges[1:]) / 2

        # Reconstructed energy binning
        self.E_p_edges = calc_bins(0.8, 12.0, n_E_p_bins)
        self.E_p = (self.E_p_edges[:-1] + self.E_p_edges[1:]) / 2

        self.IBD_Xsec_matrix: torch.tensor = torch.load(
            gcfg.data_path, map_location=gcfg.device, weights_only=True
        )
        self.IBD_Xsec_matrix_sparse = self.IBD_Xsec_matrix.to_sparse_csr()

        self.efficiency = 0.82
        self.m_p = 1.67262192369 * 1e-24  # in g
        self.m_e = 9.1093837015e-31 * 1e3  # in g
        self.N_A = 6.02214076 * 1e23
        self.M = 20 * 1e9  # in g
        self.m_H = 0.1201
        self.m_protium_aband = 0.999885
        self.N_proton = self.M * self.m_H / (self.m_p + self.m_e) * self.m_protium_aband
        self.day = day
        self.n_zero_time_calls = 0
        self.n_total_calls = 0
        self.cached_times = []
        self.uncached_times = []
        self.rebin_factor = rebin_factor
        if self.rebin_factor > 1:
            self.E_p_edges = self.E_p_edges[:: self.rebin_factor]
            self.E_p = (self.E_p_edges[:-1] + self.E_p_edges[1:]) / 2
        self.nonlinearity = NonLinearity()
        self.day_bkg = day / 11 * 12
        self.prob = SurProb_e2e()
        self.sigma_bkg = gcfg.sigma_bkg
        self.seed = 0
        self.rg = 0
        # self.IBDBin2BinErr = self.get_bin2binErrValue()
        # self.BkgBin2BinErr = self.get_bin2bin_bkg(self.day_bkg)
        # self.day_bkg_bin = None
        self.cache_monitor = CacheMonitor()  # New monitor instance
        self._cache_stats = {}  # Remove legacy stats storage
        if gcfg.IfNewBin:
            self.NewBinEdges: torch.tensor = torch.load(
                gcfg.new_edges_path, map_location=gcfg.device, weights_only=True
            )
            self.new_bin_width = self.NewBinEdges[1:] - self.NewBinEdges[:-1]
        self._finalizer = weakref.finalize(self, self.clear_all_cache)

    def calc_bins(start, end, n_bins):
        step = (end - start) / n_bins
        return torch.arange(start, end + step * 0.1, step)

    def SetRndSeed(self, seed):
        self.seed = seed
        if hasattr(torch, "get_default_device"):
            self.rg = torch.Generator(torch.get_default_device()).manual_seed(seed)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.rg = torch.Generator(device).manual_seed(seed)

    @CacheMonitor.monitor
    @lru_cache(maxsize=2)
    def get_reactor_nu_source(self, E_nu):
        # t1 = time.perf_counter()
        # Reactor parameters
        reactor_power_GW_unit = 1  # Reactor power in GW
        reactor_power_MeV_per_s = (
            reactor_power_GW_unit * 1e9 / 1.602176634 * 1e13
        )  # Convert GW to MeV/s

        # Fission fractions for each isotope
        fission_fractions = torch.tensor([0.58, 0.07, 0.30, 0.05])
        # fission_fractions = [0.564, 0.076, 0.304, 0.056]
        # Energy released per fission for each isotope in MeV
        energy_per_fission = torch.tensor([202.36, 205.99, 211.12, 214.26])

        # Step 1: Calculate the average energy released per fission for the mixture
        average_energy_per_fission = (
            fission_fractions * energy_per_fission
        ).sum()  # average_fission_energy = np.dot(fission_fractions, energy_per_fission)
        # Step 2: Calculate the total number of fissions per second
        total_fissions_per_second = (
            reactor_power_MeV_per_s / average_energy_per_fission
        )  # unit: fission/s@1GW

        # Calculate the fission count for each isotope
        fissions_per_second = total_fissions_per_second * fission_fractions

        # Load histograms using uproot
        histograms = {
            "U235": self.file["HuberMuellerFlux_U235"],
            "U238": self.file["HuberMuellerFlux_U238"],
            "Pu239": self.file["HuberMuellerFlux_Pu239"],
            "Pu241": self.file["HuberMuellerFlux_Pu241"],
        }

        # Extract histogram data
        flux_data = {}
        Enu_flux = None
        for key, hist in histograms.items():
            bin_edges = hist.axis().edges()
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Compute bin centers
            values = hist.values()

            if Enu_flux is None:
                Enu_flux = torch.tensor(bin_centers)

            flux_data[key] = torch.log10(torch.tensor(values))

        # Perform interpolation and restore exponential values
        interpolated_flux = {
            key: 10 ** torch_interp(E_nu, Enu_flux, flux_values)
            for key, flux_values in flux_data.items()
        }

        # Compute total flux integral
        d_E_nu = E_nu[1] - E_nu[0]
        integral = sum(
            interpolated_flux[key] * d_E_nu * fissions_per_second[i]
            for i, key in enumerate(["U235", "U238", "Pu239", "Pu241"])
        )

        return integral

    @CacheMonitor.monitor
    @lru_cache(maxsize=2)
    def get_reactor_nu_source_correction(self, E_nu):
        snf_hist = self.file["SNF_FluxRatio"]
        noneq_hist = self.file["NonEq_FluxRatio"]
        dnb_hist = self.file["DYBFluxBump_ratio"]

        # Compute histogram bin centers
        snf_edges = snf_hist.axis().edges()
        snf_x = (snf_edges[:-1] + snf_edges[1:]) / 2
        noneq_x = (noneq_hist.axis().edges()[:-1] + noneq_hist.axis().edges()[1:]) / 2
        dnb_x = (dnb_hist.axis().edges()[:-1] + dnb_hist.axis().edges()[1:]) / 2

        # Interpolate using torch_interp
        snf_correction = torch_interp(
            E_nu, torch.tensor(snf_x), torch.tensor(snf_hist.values())
        )

        noneq_correction = torch_interp(
            E_nu, torch.tensor(noneq_x), torch.tensor(noneq_hist.values())
        )

        f_DYB = torch_interp(E_nu, torch.tensor(dnb_x), torch.tensor(dnb_hist.values()))

        return (
            snf_correction,
            noneq_correction,
            f_DYB,
        )

    @CacheMonitor.monitor
    @lru_cache(maxsize=1000)
    def get_reactor_nu_source_corrected(self, E_nu, alpha_SNF=0, alpha_NonEq=0):
        # t1 = time.perf_counter()
        snf_correction, noneq_correction, f_DYB = self.get_reactor_nu_source_correction(
            E_nu
        )
        reactor_nu_source_corrected = (
            self.get_reactor_nu_source(E_nu)
            * (
                1
                + snf_correction * (1 + alpha_SNF)
                + noneq_correction * (1 + alpha_NonEq)
            )
            * f_DYB
        )
        # t2 = time.perf_counter()
        # print("get_reactor_nu_source_corrected Time taken: ", (t2 - t1) * 1e3, "ms")
        return reactor_nu_source_corrected

    @CacheMonitor.monitor
    @lru_cache(maxsize=10)
    def get_osc_prob(
        self,
        dmsq31=gcfg.dmsq31_NO,
        sinsq12=gcfg.sinsq12_NO,
        dmsq21=gcfg.dmsq21_NO,
        sinsq13=gcfg.sinsq13_NO,
        alpha_rho_ME=0,
        alpha_reactor1=0,
        alpha_reactor2=0,
        alpha_reactor3=0,
        alpha_reactor4=0,
        alpha_reactor5=0,
        alpha_reactor6=0,
        alpha_reactor7=0,
        alpha_reactor8=0,
        alpha_reactor9=0,
    ):
        alpha_RUCs = [
            alpha_reactor1,
            alpha_reactor2,
            alpha_reactor3,
            alpha_reactor4,
            alpha_reactor5,
            alpha_reactor6,
            alpha_reactor7,
            alpha_reactor8,
            alpha_reactor9,
        ]

        osc_prob = torch.zeros_like(self.E_nu)
        for i in range(len(rcfg.reactor_power_GW)):
            adjusted_reactor_power = rcfg.reactor_power_GW[i] * (1 + alpha_RUCs[i])
            P_osc = self.prob.surprob_Amir(
                self.E_nu, rcfg.L[i], dmsq31, sinsq12, dmsq21, sinsq13, alpha_rho_ME
            )
            scale = (
                adjusted_reactor_power * P_osc / 4 / torch.pi / rcfg.L[i] ** 2 / 100**2
            )
            osc_prob += scale  # /cm^2/s

        return osc_prob

    # TODO: move to torch
    @CacheMonitor.monitor
    @lru_cache(maxsize=10)
    def get_reactor_nu_arrival(
        self,
        dmsq31=gcfg.dmsq31_NO,
        sinsq12=gcfg.sinsq12_NO,
        dmsq21=gcfg.dmsq21_NO,
        sinsq13=gcfg.sinsq13_NO,
        alpha_rho_ME=0,
        alpha_reactor1=0,
        alpha_reactor2=0,
        alpha_reactor3=0,
        alpha_reactor4=0,
        alpha_reactor5=0,
        alpha_reactor6=0,
        alpha_reactor7=0,
        alpha_reactor8=0,
        alpha_reactor9=0,
    ):
        # t1 = time.perf_counter()
        alpha_RUCs = [
            alpha_reactor1,
            alpha_reactor2,
            alpha_reactor3,
            alpha_reactor4,
            alpha_reactor5,
            alpha_reactor6,
            alpha_reactor7,
            alpha_reactor8,
            alpha_reactor9,
        ]

        reactor_nu_arrival = torch.zeros_like(self.E_nu)
        for i in range(len(rcfg.reactor_power_GW)):
            adjusted_reactor_power = rcfg.reactor_power_GW[i] * (1 + alpha_RUCs[i])
            if gcfg.IfOsci:
                P_osc = self.prob.surprob_vacuum(
                    self.E_nu, rcfg.L[i].item(), dmsq31, sinsq12, dmsq21, sinsq13
                )
            else:
                P_osc = torch.ones_like(self.E_nu)
            scale = (
                adjusted_reactor_power * P_osc / 4 / torch.pi / rcfg.L[i] ** 2 / 100**2
            )
            reactor_nu_arrival += scale * self.get_reactor_nu_source(
                self.E_nu
            )  # /cm^2/s
        # t2 = time.perf_counter()
        # print("get_reactor_nu_arrival Time taken: ", (t2 - t1) * 1e3, "ms")
        return reactor_nu_arrival

    @CacheMonitor.monitor
    @lru_cache(maxsize=1)
    def get_E_dep_spectrum(
        self,
        dmsq31=gcfg.dmsq31_NO,
        sinsq12=gcfg.sinsq12_NO,
        dmsq21=gcfg.dmsq21_NO,
        sinsq13=gcfg.sinsq13_NO,
        alpha_rho_ME=0,
        alpha_reactor1=0,
        alpha_reactor2=0,
        alpha_reactor3=0,
        alpha_reactor4=0,
        alpha_reactor5=0,
        alpha_reactor6=0,
        alpha_reactor7=0,
        alpha_reactor8=0,
        alpha_reactor9=0,
        # osc=True,
    ):
        # t1 = time.perf_counter()
        rea_nu_arrival = (
            self.get_reactor_nu_arrival(
                dmsq31,
                sinsq12,
                dmsq21,
                sinsq13,
                alpha_rho_ME,
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
            * self.efficiency
            * self.N_proton
        )

        E_dep_spectrum = (
            self.IBD_Xsec_matrix_sparse @ rea_nu_arrival.unsqueeze(-1)
        ).squeeze(-1)

        # E_dep_spectrum = self.IBD_Xsec_matrix @ rea_nu_arrival
        # t2 = time.perf_counter()
        # print(f"get_E_dep_spectrum_6year Time taken: {(t2 - t1) * 1e3:.1f} ms")
        return E_dep_spectrum

    @CacheMonitor.monitor
    @lru_cache(maxsize=1)
    def normal_cdf_vectorized(
        self,
        alpha_Eres_a=0,
        alpha_Eres_b=0,
        alpha_Eres_c=0,
        alpha_l_pull0=0,
        alpha_l_pull1=0,
        alpha_l_pull2=0,
        alpha_l_pull3=0,
        block_size=512,  # number of columns per block
    ):
        E_vis0 = self.E_dep * self.nonlinearity(
            self.E_dep, alpha_l_pull0, alpha_l_pull1, alpha_l_pull2, alpha_l_pull3
        )
        # 2. Energy resolution parameters
        a_res = 2.614e-2 * (1 + alpha_Eres_a)
        b_res = 0.64e-2 * (1 + alpha_Eres_b)
        c_res = 1.20e-2 * (1 + alpha_Eres_c)

        sigma_Evis0 = E_vis0 * torch.sqrt(
            (a_res**2 / E_vis0) + (b_res) ** 2 + (c_res / E_vis0) ** 2
        )

        # 3. Initialize results
        n_edges = len(self.E_d_edges)
        n_ev = len(self.E_dep)

        cdf_matrix = torch.empty(
            (n_edges, n_ev), dtype=torch.float64, device=gcfg.device
        )

        sqrt2 = 2**0.5

        # 4. Block-wise computation
        for start in range(0, n_ev, block_size):
            end = min(start + block_size, n_ev)
            E_vis_block = E_vis0[start:end]
            sigma_block = sigma_Evis0[start:end]

            diff_block = (self.E_d_edges[:, None] - E_vis_block[None, :]) / (
                sigma_block[None, :] * sqrt2
            )
            cdf_matrix[:, start:end] = 0.5 * (1 + torch.erf(diff_block))
        ResponseMatrix = cdf_matrix[1:, :] - cdf_matrix[:-1, :]
        return ResponseMatrix

    def Rebin(self, in_h_y, group_size):
        # Reshape + sum to merge by group_size
        n_new_bins = in_h_y.shape[0] // group_size
        return (
            in_h_y[: n_new_bins * group_size].reshape(n_new_bins, group_size).sum(dim=1)
        )

    def block_matvec(self, mat: torch.Tensor, vec: torch.Tensor, block_size: int = 256):
        """
        稠密矩阵分块乘向量
        mat: (M, N) 稠密矩阵
        vec: (N,) 向量
        block_size: 每次处理的行数
        """
        M = mat.size(0)
        result_blocks = []
        vec = vec.unsqueeze(-1)  # (N,1)

        for start in range(0, M, block_size):
            end = min(start + block_size, M)
            block_res = mat[start:end, :] @ vec  # (block_size,1)
            result_blocks.append(block_res)

        return torch.cat(result_blocks, dim=0).squeeze(-1)  # (M,)

    @CacheMonitor.monitor
    @lru_cache(maxsize=1000)
    def get_E_p_spectrum(
        self,
        dmsq31=gcfg.dmsq31_NO,
        sinsq12=gcfg.sinsq12_NO,
        dmsq21=gcfg.dmsq21_NO,
        sinsq13=gcfg.sinsq13_NO,
        alpha_l_pull0=0,
        alpha_l_pull1=0,
        alpha_l_pull2=0,
        alpha_l_pull3=0,
        alpha_Eres_a=0,
        alpha_Eres_b=0,
        alpha_Eres_c=0,
        alpha_rho_ME=0,
        alpha_reactor1=0,
        alpha_reactor2=0,
        alpha_reactor3=0,
        alpha_reactor4=0,
        alpha_reactor5=0,
        alpha_reactor6=0,
        alpha_reactor7=0,
        alpha_reactor8=0,
        alpha_reactor9=0,
    ):
        # t0 = time.perf_counter()
        t1 = time.perf_counter()
        ResponseMatrix = self.normal_cdf_vectorized(
            alpha_Eres_a,
            alpha_Eres_b,
            alpha_Eres_c,
            alpha_l_pull0,
            alpha_l_pull1,
            alpha_l_pull2,
            alpha_l_pull3,
        )

        edep = (
            self.get_E_dep_spectrum(
                dmsq31,
                sinsq12,
                dmsq21,
                sinsq13,
                alpha_rho_ME,
                alpha_reactor1,
                alpha_reactor2,
                alpha_reactor3,
                alpha_reactor4,
                alpha_reactor5,
                alpha_reactor6,
                alpha_reactor7,
                alpha_reactor8,
                alpha_reactor9,
                # osc,
            )
            * self.day
            * 3600
            * 24
        )
        N_Ep_vals = self.block_matvec(ResponseMatrix, edep, block_size=512)
        # ResponseMatrix = ResponseMatrix.to_sparse_csr()
        # N_Ep_vals = torch.sparse.mm(ResponseMatrix, edep.unsqueeze(1)).squeeze(1)

        N_Ep_vals = self.Rebin(N_Ep_vals, int(len(N_Ep_vals) // len(self.E_p)))
        # ResponseMatrix_sparse = ResponseMatrix.to_sparse_csr()  # compressed storage
        # N_Ep_vals = ResponseMatrix_sparse @ edep
        # t2 = time.perf_counter()
        # dt = t2 - t1
        # dt_ms = dt * 1e3  # milliseconds
        # dt_rounded = round(dt_ms, 1)

        # self.n_total_calls += 1
        # if dt_rounded == 0.0:
        #     self.n_zero_time_calls += 1
        #     self.cached_times.append(dt_ms)
        # else:
        #     self.uncached_times.append(dt_ms)
        # print(f"get_E_p_spectrum Time taken: {dt_ms:.3f} ms")
        if self.seed > 0:
            N_Ep_vals = self.add_shape_unc_2ibd(N_Ep_vals)
        return N_Ep_vals

    def rebin_histogram(self, bin_values, bin_edges, rebin_factor=None):
        """通用直方图重分组方法"""
        rebin_factor = rebin_factor
        valid_length = len(bin_values) // rebin_factor * rebin_factor
        rebinned_values = (
            bin_values[:valid_length].reshape(-1, rebin_factor).sum(axis=1)
        )
        rebinned_edges = bin_edges[::rebin_factor]
        return rebinned_values, rebinned_edges

    # get background histgram per day
    @CacheMonitor.monitor
    @lru_cache(maxsize=2)
    def get_background_spectrum_pdf(self):
        # Extract background histograms from the ROOT file

        pdfs = []
        for i, tree_name in enumerate(gcfg.m_background_trees):
            histogram = self.file[tree_name]
            if histogram:
                bin_values = histogram.values()
                bin_values *= gcfg.m_bkg_rate[i] / bin_values.sum()
                bin_edges = histogram.axis().edges()
                bin_edges = torch.tensor(bin_edges, dtype=torch.float64)
                bin_values = torch.tensor(bin_values, dtype=torch.float64)
                bin_values = self.RebinHist(bin_edges, bin_values, self.E_p_edges)

                if self.rebin_factor > 1:
                    rebinned_values, _ = self.rebin_histogram(
                        bin_values, bin_edges, self.rebin_factor
                    )
                    pdfs.append(torch.tensor(rebinned_values, dtype=torch.float64))
                else:
                    pdfs.append(torch.tensor(bin_values, dtype=torch.float64))

        return torch.stack(pdfs)

    @CacheMonitor.monitor
    @lru_cache(maxsize=1000)
    def get_background_spectrum(
        self,
        alpha_bkg0=0,
        alpha_bkg1=0,
        alpha_bkg2=0,
        alpha_bkg3=0,
        alpha_bkg4=0,
        alpha_bkg5=0,
        alpha_bkg6=0,
    ):
        # Extract background histograms from the ROOT file
        alphas_bkg = [
            alpha_bkg0,
            alpha_bkg1,
            alpha_bkg2,
            alpha_bkg3,
            alpha_bkg4,
            alpha_bkg5,
            alpha_bkg6,
        ]

        total_background = torch.zeros(len(self.E_p))
        pdfs = self.get_background_spectrum_pdf()
        for i, tree_name in enumerate(gcfg.m_background_trees):
            bin_vals = pdfs[i].clone()
            bin_vals *= 1 + alphas_bkg[i]
            if self.seed > 0:
                bin_vals = self.add_shape_unc_2bkg(i, bin_vals)

            total_background += bin_vals
        total_background *= self.day_bkg
        return total_background

    def get_divided_background_spectrum(
        self,
        alpha_bkg0=0,
        alpha_bkg1=0,
        alpha_bkg2=0,
        alpha_bkg3=0,
        alpha_bkg4=0,
        alpha_bkg5=0,
        alpha_bkg6=0,
    ):
        alphas_bkg = torch.tensor(
            [
                alpha_bkg0,
                alpha_bkg1,
                alpha_bkg2,
                alpha_bkg3,
                alpha_bkg4,
                alpha_bkg5,
                alpha_bkg6,
            ]
        ).unsqueeze(1)  # (7, 1)
        pdfs = self.get_background_spectrum_pdf()
        divided_bkg_spec = pdfs * self.day_bkg * (1 + alphas_bkg)
        return divided_bkg_spec

    @CacheMonitor.monitor
    @lru_cache(maxsize=1)
    def get_bin2bin_bkg_abs(self, day_bkg=1):
        bin2bin_bkg = torch.zeros((len(gcfg.m_background_trees), len(self.E_p)))

        for k, tree_name in enumerate(gcfg.m_background_trees):
            if tree_name in self.file:
                histogram = self.file[tree_name]
                original_edges = torch.tensor(histogram.axis().edges())
                original_values = torch.tensor(histogram.values())

                # Dynamically rebin to the current binning
                rebinned_values = self.RebinHist(
                    original_edges, original_values, self.E_p_edges
                )

                bin2bin_bkg[k] = (
                    rebinned_values
                    * rebinned_values
                    * self.sigma_bkg[k] ** 2
                    * day_bkg**2
                )
        return bin2bin_bkg

    @CacheMonitor.monitor
    @lru_cache(maxsize=1)
    def add_shape_unc_2bkg(self, ibkg, bkg_spectrum):
        # Adjust bin width based on rebinning state
        data_bin_width = (
            gcfg.DataBinWidth * self.rebin_factor
            if self.rebin_factor > 1
            else gcfg.DataBinWidth
        )
        sigma_new = (
            torch.sqrt(torch.tensor(0.036 / data_bin_width)) * gcfg.sigma_bkg[ibkg]
        )
        weights = torch.normal(
            0, 1, (len(bkg_spectrum),), generator=self.rg, device=gcfg.device
        )
        return bkg_spectrum * (1 + sigma_new * weights)

    @CacheMonitor.monitor
    @lru_cache(maxsize=1)
    def add_shape_unc_2ibd(self, signal_spectrum):
        data_bin_width = (
            gcfg.DataBinWidth * self.rebin_factor
            if self.rebin_factor > 1
            else gcfg.DataBinWidth
        )
        sigma_new = (
            torch.sqrt(torch.tensor(0.02 / data_bin_width)) * self.get_bin2bin_sig()
        )
        weights = torch.normal(
            0, 1, (len(signal_spectrum),), generator=self.rg, device=gcfg.device
        )
        return signal_spectrum * (1 + sigma_new * weights)

    @CacheMonitor.monitor
    @lru_cache(maxsize=1000)
    def get_signal_plus_background(
        self,
        dmsq31=gcfg.dmsq31_NO,
        sinsq12=gcfg.sinsq12_NO,
        dmsq21=gcfg.dmsq21_NO,
        sinsq13=gcfg.sinsq13_NO,
        alpha_Eres_a=0,
        alpha_Eres_b=0,
        alpha_Eres_c=0,
        alpha_rho_ME=0,
        alpha_bkg0=0,
        alpha_bkg1=0,
        alpha_bkg2=0,
        alpha_bkg3=0,
        alpha_bkg4=0,
        alpha_bkg5=0,
        alpha_bkg6=0,
        alpha_reactor1=0,
        alpha_reactor2=0,
        alpha_reactor3=0,
        alpha_reactor4=0,
        alpha_reactor5=0,
        alpha_reactor6=0,
        alpha_reactor7=0,
        alpha_reactor8=0,
        alpha_reactor9=0,
        alpha_l_pull0=0,
        alpha_l_pull1=0,
        alpha_l_pull2=0,
        alpha_l_pull3=0,
        alpha_RC=0,
        alpha_D=0,
    ):
        """
        获取信号加本底的叠加能谱。

        参数：
        - 各本底拟合参数：alpha_bkg_AccBkgHistogramAD 等 7 个本底修正参数

        返回：
        - signal_plus_background: 信号和本底叠加后的总能谱
        """
        t1 = time.perf_counter()
        signal_spectrum = self.get_E_p_spectrum(
            dmsq31,
            sinsq12,
            dmsq21,
            sinsq13,
            alpha_l_pull0,
            alpha_l_pull1,
            alpha_l_pull2,
            alpha_l_pull3,
            alpha_Eres_a,
            alpha_Eres_b,
            alpha_Eres_c,
            alpha_rho_ME,
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

        background_spectrum = self.get_background_spectrum(
            alpha_bkg0,
            alpha_bkg1,
            alpha_bkg2,
            alpha_bkg3,
            alpha_bkg4,
            alpha_bkg5,
            alpha_bkg6,
        )

        Pull_2_T = 1 + alpha_RC + alpha_D
        signal_plus_background = signal_spectrum * Pull_2_T + background_spectrum
        if self.seed:
            signal_plus_background = torch.where(
                signal_plus_background < 0, 0, signal_plus_background
            )
            signal_plus_background = torch.poisson(
                signal_plus_background, generator=self.rg
            )
        # t2 = time.perf_counter()
        # print(f"get_signal_plus_background Time taken: {(t2 - t1) * 1e3:.1f} ms")
        return signal_plus_background

    # rebin histogram
    # input: input histogram edges, input histogram y, output histogram edges
    def RebinHist(self, in_h_lowedge, in_h_y, out_h_edge):
        in_bin_width = in_h_lowedge[1] - in_h_lowedge[0]
        low_idx = (
            (out_h_edge[:-1] + 0.5 * in_bin_width - in_h_lowedge[0]) // in_bin_width
        ).long()
        high_idx = (
            (out_h_edge[1:] - 0.5 * in_bin_width - in_h_lowedge[0]) // in_bin_width
        ).long()

        equal_mask = low_idx == high_idx
        out_h_y = torch.zeros(
            len(out_h_edge) - 1, dtype=in_h_y.dtype, device=in_h_y.device
        )
        out_h_y[equal_mask] = in_h_y[low_idx[equal_mask]]
        not_equal_mask = ~equal_mask
        for i in torch.nonzero(not_equal_mask).flatten():
            # Fix: include high_idx[i]
            out_h_y[i] = in_h_y[low_idx[i] : high_idx[i] + 1].sum()

        return out_h_y

    @CacheMonitor.monitor
    @lru_cache(maxsize=1)
    def get_bin2bin_sig(self):
        return gcfg.gsigma_signal_b2b

    def clear_all_cache(self):
        """自动清理所有缓存方法"""
        for attr in dir(self):
            try:
                method = getattr(self, attr)
                if callable(method) and hasattr(method, "cache_clear"):
                    method.cache_clear()
            except Exception as e:  # Handle attribute access exceptions
                print(f"Error clearing cache for {attr}: {str(e)}")
        if hasattr(self.prob, "clear_cache"):
            self.prob.clear_cache()  # Clear SurProb_e2e cache
        # Additionally clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # def __del__(self):
    #     """Cleanup on destructor call"""
    #     self.clear_all_cache()
