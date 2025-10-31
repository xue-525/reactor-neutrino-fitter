import numpy as np
from scipy import integrate


class InverseBetaDecayXS:
    """Compute IBD cross section from Strumia & Vissani (2003).
    Precise quasielastic neutrino/nucleon cross-section,
    doi = "10.1016/S0370-2693(03)00616-6",
    journal = "Phys. Lett. B",
    """

    def __init__(self):
        # --- Physical constants (MeV units) ---
        self.gA0 = -1.2762
        self.M_p = 938.272046
        self.M_n = 939.565379
        self.M = (self.M_p + self.M_n) / 2
        self.M_pi = 139.57018
        self.Me = 0.510998928
        self.Delta = self.M_n - self.M_p  # neutron–proton mass difference

        self.G_F = 1.1663787e-11  # Fermi coupling constant [MeV^-2]
        self.Vud = 0.97401117  # CKM matrix element cos(θ_C)
        self.alpha = 1.0 / 137.036  # fine-structure constant

        self.kappa_p = 1.7928473
        self.kappa_n = -1.913042
        self.xi = self.kappa_p - self.kappa_n

        # Axial and vector form-factor scales
        self.M_A2 = (1.026e3) ** 2  # (MeV)^2
        self.M_V2 = (0.840e3) ** 2  # (MeV)^2

        # Derived constants for normalization
        tau_n = 879.4 * 1.519e21  # neutron lifetime in MeV^-1
        f_R = 1.7152
        f, g = 1.0, 1.26
        sigma_tot = 2 * np.pi**2 / (self.Me**5 * f_R * tau_n)
        self.sigma_0 = sigma_tot / (f**2 + 3 * g**2)

    # ---------------- Lorentz invariants ----------------
    def s_invariant(self, E_nu):
        return 2 * self.M_p * E_nu + self.M_p**2

    def s_minus_u(self, E_nu, E_e):
        return 2 * self.M_p * (E_nu + E_e) - self.Me**2

    def t_invariant(self, E_nu, E_e):
        return self.M_n**2 - self.M_p**2 - 2 * self.M_p * (E_nu - E_e)

    # ---------------- Form factors ----------------
    def axial_ff(self, E_nu, E_e):
        t = self.t_invariant(E_nu, E_e)
        return self.gA0 / (1 - t / self.M_A2) ** 2

    def induced_ff(self, E_nu, E_e):
        t = self.t_invariant(E_nu, E_e)
        return 2 * self.M**2 * self.axial_ff(E_nu, E_e) / (self.M_pi**2 - t)

    def vector_ff(self, E_nu, E_e):
        t = self.t_invariant(E_nu, E_e)
        denom = (1 - t / (4 * self.M**2)) * (1 - t / self.M_V2) ** 2
        return (1 - (1 + self.xi) * t / (4 * self.M**2)) / denom

    def weakmag_ff(self, E_nu, E_e):
        t = self.t_invariant(E_nu, E_e)
        denom = (1 - t / (4 * self.M**2)) * (1 - t / self.M_V2) ** 2
        return self.xi / denom

    # ---------------- A, B, C coefficients ----------------
    def A_term(self, E_nu, E_e):
        t = self.t_invariant(E_nu, E_e)
        f1, f2 = self.vector_ff(E_nu, E_e), self.weakmag_ff(E_nu, E_e)
        g1, g2 = self.axial_ff(E_nu, E_e), self.induced_ff(E_nu, E_e)

        part1 = (t - self.Me**2) * (
            4 * (abs(f1) ** 2) * (4 * self.M**2 + t + self.Me**2)
            + 4 * (abs(g1) ** 2) * (-4 * self.M**2 + t + self.Me**2)
            + abs(f2) ** 2 * (t**2 / self.M**2 + 4 * t + 4 * self.Me**2)
            + 4 * self.Me**2 * t * abs(g2) ** 2 / self.M**2
            + 8 * f1 * f2 * (2 * t + self.Me**2)
            + 16 * self.Me**2 * g1 * g2
        )

        part2 = self.Delta**2 * (
            (4 * abs(f1) ** 2 + t * abs(f2) ** 2 / self.M**2)
            * (4 * self.M**2 + t - self.Me**2)
            + 4 * abs(g1) ** 2 * (4 * self.M**2 - t + self.Me**2)
            + 4 * self.Me**2 * abs(g2) ** 2 * (t - self.Me**2) / self.M**2
            + 8 * f1 * f2 * (2 * t - self.Me**2)
            + 16 * self.Me**2 * g1 * g2
        )

        part3 = 32 * self.Me**2 * self.M * self.Delta * g1 * (f1 + f2)
        return (part1 - part2 - part3) / 16.0

    def B_term(self, E_nu, E_e):
        t = self.t_invariant(E_nu, E_e)
        f1, f2 = self.vector_ff(E_nu, E_e), self.weakmag_ff(E_nu, E_e)
        g1, g2 = self.axial_ff(E_nu, E_e), self.induced_ff(E_nu, E_e)

        term = (
            16 * t * g1 * (f1 + f2)
            + 4
            * self.Me**2
            * self.Delta
            * (abs(f2) ** 2 + f1 * f2 + 2 * g1 * g2)
            / self.M
        )
        return term / 16.0

    def C_term(self, E_nu, E_e):
        t = self.t_invariant(E_nu, E_e)
        f1, f2 = self.vector_ff(E_nu, E_e), self.weakmag_ff(E_nu, E_e)
        g1 = self.axial_ff(E_nu, E_e)
        val = 4 * (abs(f1) ** 2 + abs(g1) ** 2) - t * abs(f2) ** 2 / self.M**2
        return val / 16.0

    # ---------------- Differential & total cross sections ----------------
    def matrix_element_sq(self, E_nu, E_e):
        s_u = self.s_minus_u(E_nu, E_e)
        return (
            self.A_term(E_nu, E_e)
            - s_u * self.B_term(E_nu, E_e)
            + s_u**2 * self.C_term(E_nu, E_e)
        )

    def dσ_dt(self, E_nu, E_e):
        s = self.s_invariant(E_nu)
        prefactor = self.G_F**2 * self.Vud**2 / (2 * np.pi * (s - self.M_p**2) ** 2)
        conv = (1.0 / (5.068e10)) ** 2  # convert MeV^-2 → cm^2
        return prefactor * self.matrix_element_sq(E_nu, E_e) * conv

    def dσ_dEe(self, E_nu, E_e):
        return 2 * self.M_p * self.dσ_dt(E_nu, E_e)

    def dσ_dcos(self, E_nu, cos_theta):
        """Differential cross section dσ/dcosθ without radiative correction."""
        eps = E_nu / self.M_p
        kappa = (1 + eps) ** 2 - (eps * cos_theta) ** 2

        if (E_nu - self.Delta) ** 2 - self.Me**2 * kappa <= 0 or kappa <= 0:
            return 0.0

        E_e = (
            (E_nu - self.Delta) * (1 + eps)
            + eps * cos_theta * np.sqrt((E_nu - self.Delta) ** 2 - self.Me**2 * kappa)
        ) / kappa

        if E_e < self.Me:
            return 0.0

        p_e = np.sqrt(E_e**2 - self.Me**2)
        jacobian = p_e * eps / (1 + eps * (1 - E_e * cos_theta / p_e))
        return jacobian * self.dσ_dEe(E_nu, E_e)

    def dσ_dcos_with_RC(self, cos_theta, E_nu):
        """Include radiative correction (Strumia & Vissani, Eq. 25)."""
        eps = E_nu / self.M_p
        kappa = (1 + eps) ** 2 - (eps * cos_theta) ** 2
        if (E_nu - self.Delta) ** 2 - self.Me**2 * kappa <= 0 or kappa <= 0:
            return 0.0

        E_e = (
            (E_nu - self.Delta) * (1 + eps)
            + eps * cos_theta * np.sqrt((E_nu - self.Delta) ** 2 - self.Me**2 * kappa)
        ) / kappa

        if E_e <= 0:
            return 0.0

        rc_factor = 1 + self.alpha / np.pi * (
            6.00 + 1.5 * np.log(self.M_p / (2 * E_e)) + 1.2 * (self.Me / E_e) ** 1.5
        )
        return self.dσ_dcos(cos_theta, E_nu) * rc_factor

    def total_xsec(self, E_nu):
        val, _ = integrate.quad(self.dσ_dcos, -1, 1, args=(E_nu,))
        return val

    def total_xsec_RC(self, E_nu):
        val, _ = integrate.quad(self.dσ_dcos_with_RC, -1, 1, args=(E_nu,))
        return val

    def total_xsec_RC_array(self, E_nu_array):
        return np.array([self.total_xsec_RC(E) for E in E_nu_array])

    def GetE_e(self, E_nu, Cos_theta):
        E_e = 0
        M_proton = self.M_p / 1e3  # in GeV
        M_neutron = self.M_n / 1e3  # in GeV
        M_electron = self.Me / 1e3  # in GeV
        E_Lower = 1.80607
        M_e_sq = M_electron * M_electron
        M_p_sq = M_proton * M_proton
        M_n_sq = M_neutron * M_neutron
        delta_mr = (M_n_sq - M_p_sq - M_e_sq) / (2.0 * M_proton)
        if E_nu > E_Lower:
            E_nu = E_nu / 1e3  # to GeV
            epsilon_nu2p = E_nu / M_proton  # ratio E_nu/m_P
            kappa_ec = pow(1 + epsilon_nu2p, 2) - pow(epsilon_nu2p * Cos_theta, 2)
            E_nuMD = E_nu - delta_mr  # E_nu -delta
            E_e = (
                E_nuMD * (1.0 + epsilon_nu2p)
                + epsilon_nu2p
                * Cos_theta
                * np.sqrt(E_nuMD * E_nuMD - M_e_sq * kappa_ec)
            ) / kappa_ec
            if E_e < M_electron:
                return 0
            # printf("E_e?: %e\n", E_e);
            E_e = E_e * 1e3  # back to MeV

        return E_e


if __name__ == "__main__":
    ibd = InverseBetaDecayXS()
    E_test = np.linspace(2, 8, 10)
    print(ibd.total_xsec_RC_array(E_test))
