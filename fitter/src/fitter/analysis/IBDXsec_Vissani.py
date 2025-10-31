# Differential cross-section data has been generated and can be used directly
import numpy as np
import sys
import os

# Get directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Compute the module directory
module_dir = os.path.join(current_dir, "../physics")
# Add the module directory to sys.path
sys.path.append(module_dir)

# Now modules can be imported normally
from ibdxsec_Vissani import InverseBetaDecayXS
import tqdm
import numba as nb


@nb.njit
def get_index(T_1, low0, binwid, Me):
    return int(np.floor((T_1 + 2 * Me - low0) / binwid))


def getIBD_Xsec(
    out_E_nu_low=1.8,  # MeV
    out_E_nu_up=13.0,  # MeV
    n_E_nu_bins=5600,  # total number of bins
    out_E_dep_low=0.8,  # MeV
    out_E_dep_up=12.0,  # MeV
    n_E_dep_bins=5600,  # total number of bins
):
    def calc_bins(start, end, n_bins):
        step = (end - start) / n_bins
        return np.arange(
            start, end + step * 0.1, step
        )  # + step*0.1 ensures the endpoint is included

    E_nu_edges = calc_bins(out_E_nu_low, out_E_nu_up, n_E_nu_bins)
    E_dep_bins = calc_bins(out_E_dep_low, out_E_dep_up, n_E_dep_bins)

    # Compute actual bin width for filenames
    out_E_nu_binwid = (out_E_nu_up - out_E_nu_low) / n_E_nu_bins
    out_E_dep_binwid = (out_E_dep_up - out_E_dep_low) / n_E_dep_bins
    # IBDXsec_Vissani = IBDXSec_VogelBeacom_DYB()
    IBDXsec_Vissani = InverseBetaDecayXS()

    nsplit_costheta = 20
    costheta_values = np.linspace(-1, 1, nsplit_costheta + 1)
    costheta_values_middle = (costheta_values[:-1] + costheta_values[1:]) / 2
    costheta_binwidth = 2.0 / nsplit_costheta

    E_dep_bin_middles = (E_dep_bins[:-1] + E_dep_bins[1:]) / 2
    E_nu_low = E_nu_edges[:-1]
    nfine_split_E_nu = 10

    # Store integrated cross section at each E_nu(i) (2D array)
    IBD_Xsec_matrix = np.zeros(
        (len(E_dep_bin_middles), len(E_nu_low))
    )  # Create a 2D array with shape (number of E_dep bins, number of E_nu bins)
    # For each E_nu(i), integrate and save each differential cross-section value
    edep_low0 = E_dep_bins[0]
    with tqdm.tqdm(total=len(E_nu_low)) as pbar:
        for i in range(len(E_nu_low)):
            E_nu_i = E_nu_low[i]
            d_E_nu = E_nu_low[1] - E_nu_low[0]
            # Compute the differential cross section for each costheta and save to the 2D array
            for j, costheta in enumerate(costheta_values_middle):
                for k in range(nfine_split_E_nu):
                    E_nu_k = E_nu_i + d_E_nu * (k + 0.5) / nfine_split_E_nu
                    # T_1 = Ee_1(costheta,E_nu_k)-me
                    T_1 = IBDXsec_Vissani.GetE_e(E_nu_k, costheta) - IBDXsec_Vissani.Me
                    edep_l = get_index(
                        T_1, edep_low0, out_E_dep_binwid, IBDXsec_Vissani.Me
                    )
                    if edep_l < 0 or edep_l >= len(E_dep_bin_middles):
                        continue
                    pbar.set_postfix(
                        E_nu=E_nu_k,
                        cos=costheta,
                        T_1=T_1,
                        T_0=IBDXsec_Vissani.GetE_e(E_nu_k, 0) - IBDXsec_Vissani.Me,
                    )
                    # Within a costheta bin the differential cross section is constant; multiply by bin width for the bin total; E_nu is uniformly sub-binned and averaged by dividing by the number of sub-bins
                    IBD_Xsec_matrix[edep_l, i] += (
                        IBDXsec_Vissani.dσ_dcos(E_nu_k, costheta)
                        * costheta_binwidth
                        / nfine_split_E_nu
                    )
                    pbar.update(1)
    import torch

    tensor_data = torch.from_numpy(IBD_Xsec_matrix)
    # torch.save(
    #     tensor_data,
    #     f"../../../data/outputs/IBDXsec_DYB_Edep_Vissani_matrix_enu{out_E_nu_binwid * 1e3:.0f}keV_edep{out_E_dep_binwid * 1e3:.0f}keV.pt",
    # )
    output_path = (
        f"../../../data/outputs/IBDXsec_Vissani_matrix_enu{n_E_nu_bins}_test.pt"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(
        tensor_data,
        output_path,
    )


# getIBD_Xsec(1.8,13.0,0.002,0.8,12.0,0.002)
# getIBD_Xsec(1.8, 13.0, 0.005, 0.8, 12.0, 0.005)
# getIBD_Xsec(
#     out_E_nu_low=1.8,
#     out_E_nu_up=13.0,
#     n_E_nu_bins=1000,
#     out_E_dep_low=0.8,
#     out_E_dep_up=12.0,
#     n_E_dep_bins=1000,
# )
# Test different bin configurations
# bin_list = [10, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
bin_list = [5600]

for bins in bin_list:
    print(f"\n>>> Running for n_E_nu_bins = n_E_dep_bins = {bins}")

    getIBD_Xsec(n_E_nu_bins=bins, n_E_dep_bins=bins)
