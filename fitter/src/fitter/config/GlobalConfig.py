import torch

""" torch related """
device = "cpu"  # cpu or cuda
monitoring = False
enable_caching = True  # enable caching
""" Input Paths"""
file_path = "./fitter/data/data_example.root"
data_path = "./fitter/data/outputs/IBDXsec_matrix_enu5600_test.pt"

new_edges_path = "./fitter/src/fitter/config/EnergyEdges.pt"


""" test-statistic """
test_statistic = 0  # 0:CNP, 1:Pearson, 2:Pearson+log|V|

end_bin = -175  # 9 MeV(-150); 8.5 MeV(-175) used for log|V|
start_bin = 0  # start bin for analysis


""" presumed reactor correlated uncertainty"""
m_sigma_ReaCorr = 0.02
"""presumed reactor uncorrelated uncertainty"""
msigma_ReaUnC = [
    0.008,
    0.008,
    0.008,
    0.008,
    0.008,
    0.008,
    0.008,
    0.008,
    0.008,
]

"""presumed detector correlation uncertainty"""
gsigma_DetectorCorr = 0.01
"""background trees"""
m_background_trees = [
    "bkg0",
    "bkg1",
    "bkg2",
    "bkg3",
    "bkg4",
    "bkg5",
    "bkg6",
]
"""presumed background rate"""
m_bkg_rate = [0.8, 0.1, 0.8, 0.05, 1.2, 1.0, 0.16]
"""presumed background rate uncertainty"""
m_sigma_bkg_rate = [0.01, 1, 0.2, 0.5, 0.3, 0.02, 0.5]
"""presumed background shape uncertainty"""
sigma_bkg = torch.tensor([0, 0.2, 0.1, 0.5, 0.05, 0.05, 0.5])

"""presumed nonlinearity relative scale factor"""
NLRelativeScaleFactor = [1, 1, 1, 1]

# energy resolution
"""Abusleme A, et al. Calibration Strategy of the JUNO Experiment [J/OL]. JHEP, 2021, 03:  004. DOI: 10.1007/JHEP03(2021)004."""
gsigma_Eres_a = 0.0073
gsigma_Eres_b = 0.0138
gsigma_Eres_c = 0.0262


bin2bin_width = 0.02  # signal shape uncertainty raw bin width
DataBinWidth = 0.02  # data bin width

"""DYB sinsq13 values"""
sinsq13_0 = 2.19e-2  # pull center
sigma_sinsq13 = 0.0007

"""pdg dmsq31 values(https://pdg.lbl.gov/)"""
dmsq31_0_NO = 2.5303e-3
dmsq31_0_IO = -0.0024537
"""pdg precision"""
grsigma_dmsq31 = 0.0118
"""presume signb2b uncertainty"""
gsigma_signal_b2b = 0.0035  # 0.35%/20keV

""" pdg values(https://pdg.lbl.gov/)
    Navas, S. and others, Review of particle physics, Phys. Rev. D, 110, 030001 (2024)
"""
""" True Values """
# NO true values
sinsq12_NO = 0.307
sinsq13_NO = 2.19e-2
dmsq21_NO = 7.53e-5
dmsq32_NO = 2.455e-3
dmsq31_NO = dmsq32_NO + dmsq21_NO

# IO true values
sinsq12_IO = 0.307
sinsq13_IO = 2.19e-2
dmsq21_IO = 7.53e-5
dmsq32_IO = -0.002529
dmsq31_IO = dmsq32_IO + dmsq21_IO

"""Flag"""
IfConstrainSinsq13 = True
IfConstrainDmsq31 = False
IfDYB2dContour = False
IfConstrainDmsq21 = False
IfConstrainRC = True
IfConstrainRUCs = True
IfConstrainD = True
IfConstrainBkg = True
IfConstrainNL = True
IfConstrainEres = True
IfConstrainRhoME = True
IfConstrainSNF = True
IfConstrainNonEq = True

IfOsci = True
IfNewBin = False
IfScaleIBDShpUnc = True

""" Functions """


def s2t13_sq_to_sinsq13(s2t13_sq):
    return 0.5 * (1.0 - torch.sqrt(1.0 - s2t13_sq))
