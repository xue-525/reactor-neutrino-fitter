"""
L-BFGS fitter benchmark: compare L-BFGS with iMinuit on reactor neutrino fitting.
Entry point for validation per claude.md Section 12.1.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from src.fitter.fitting_frame.torch_helper import fit_lbfgs
from loguru import logger
import time
from src.fitter.analysis.fitter import Fitter
from src.fitter.config import GlobalConfig as gcfg

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

OSC_PARAM_NAMES = ["dmsq31", "sinsq12", "dmsq21", "sinsq13"]


def create_juno_syst(seed=50):
    """Create and initialize JunoSyst with toy observed spectrum."""
    gcfg.test_statistic = 0
    year = 50 / 365.25
    juno_syst = Fitter(
        year,
        n_E_nu_bins=5600,
        n_E_dep_bins=5600,
        n_E_d_bins=560,
        n_E_p_bins=560,
    )
    obs_spectrum = juno_syst.get_obs_spectrum(seed=seed)
    logger.info(f"Observed spectrum: {len(obs_spectrum)} bins, {obs_spectrum.sum():.0f} events")

    try:
        truth = juno_syst.get_truth_params(as_dict=True, print_out=False)
        logger.info(f"Truth parameters: {truth}")
    except Exception:
        pass

    return juno_syst


def run_minuit_fit(juno_syst):
    """Run iMinuit fit for validation baseline."""
    init_vals = juno_syst.fit_para_init.detach().cpu().numpy()
    names = list(juno_syst.fit_para_names)

    t0 = time.perf_counter()
    m = Minuit(juno_syst.chi2, init_vals, name=names)
    m.tol = 0.01
    m.migrad()
    elapsed = time.perf_counter() - t0

    fmin = m.fmin
    logger.info(f"[Minuit] Valid: {fmin.is_valid}, fval: {fmin.fval:.6e}, EDM: {fmin.edm:.6e}")
    logger.info(f"[Minuit] nfcn: {fmin.nfcn}, ngrad: {fmin.ngrad}, time: {elapsed:.4f}s")
    for name in OSC_PARAM_NAMES:
        logger.info(f"[Minuit]   {name} = {m.values[name]:.6e}")

    values = {name: m.values[name] for name in names}
    return {
        "values": values,
        "fval": fmin.fval,
        "edm": fmin.edm,
        "valid": fmin.is_valid,
        "nfcn": fmin.nfcn,
        "ngrad": fmin.ngrad,
        "time": elapsed,
    }


def run_lbfgs_fit(juno_syst, edm_tolerance=1e-5, max_iter=200):
    """Run L-BFGS fit."""
    t0 = time.perf_counter()
    x, y, t, lr, model, edm_values = fit_lbfgs(
        juno_syst,
        initial_params=None,
        max_iter=max_iter,
        lr=1.0,
        history_size=10,
        line_search_fn="strong_wolfe",
        edm_tolerance=edm_tolerance,
        edm_method="lbfgs",
        use_parameter_scaling=True,
    )
    elapsed = time.perf_counter() - t0

    osc_params = model.get_oscillation_params()
    all_params = model.get_fitted_params()
    final_edm = edm_values[-1] if edm_values else None

    logger.info(f"[L-BFGS] Steps: {len(x)}, fval: {y[-1]:.6e}, EDM: {final_edm:.6e}")
    logger.info(f"[L-BFGS] Time: {elapsed:.4f}s")
    for name in OSC_PARAM_NAMES:
        logger.info(f"[L-BFGS]   {name} = {osc_params[name]:.6e}")

    return {
        "x": x,
        "y": y,
        "t": t,
        "lr": lr,
        "model": model,
        "edm_values": edm_values,
        "osc_params": osc_params,
        "all_params": all_params,
        "fval": y[-1],
        "edm": final_edm,
        "time": elapsed,
        "n_steps": len(x),
    }


def compare_lbfgs_vs_minuit():
    """Main comparison: L-BFGS vs iMinuit."""
    logger.info("=" * 60)
    logger.info("L-BFGS vs iMinuit Comparison")
    logger.info("=" * 60)

    juno_syst = create_juno_syst(seed=1)

    # Run both fitters
    lbfgs_result = run_lbfgs_fit(juno_syst, edm_tolerance=1e-5, max_iter=200)
    minuit_result = run_minuit_fit(juno_syst)

    # Print comparison table
    logger.info("\n" + "=" * 60)
    logger.info("Comparison Summary")
    logger.info("=" * 60)
    logger.info(f"{'':20s} {'L-BFGS':>15s} {'Minuit':>15s} {'Diff':>12s}")
    logger.info("-" * 62)

    # Chi2
    diff_chi2 = lbfgs_result["fval"] - minuit_result["fval"]
    logger.info(f"{'Final chi2':20s} {lbfgs_result['fval']:15.6e} {minuit_result['fval']:15.6e} {diff_chi2:12.2e}")

    # EDM
    logger.info(f"{'Final EDM':20s} {lbfgs_result['edm']:15.6e} {minuit_result['edm']:15.6e}")

    # Oscillation parameters
    for name in OSC_PARAM_NAMES:
        v_lbfgs = lbfgs_result["osc_params"][name]
        v_minuit = minuit_result["values"][name]
        rel_diff = abs(v_lbfgs - v_minuit) / abs(v_minuit) * 100 if v_minuit != 0 else 0
        logger.info(f"{name:20s} {v_lbfgs:15.6e} {v_minuit:15.6e} {rel_diff:11.4f}%")

    # Timing
    logger.info(f"{'Time (s)':20s} {lbfgs_result['time']:15.4f} {minuit_result['time']:15.4f}")
    logger.info(f"{'Steps/nfcn':20s} {lbfgs_result['n_steps']:15d} {minuit_result['nfcn']:15d}")

    # Consistency check
    if abs(diff_chi2) > 0.1:
        logger.warning(f"Chi2 difference ({diff_chi2:.2e}) exceeds 0.1 — results may be inconsistent!")
    else:
        logger.info("Chi2 values are consistent between L-BFGS and Minuit.")

    # Generate plots
    plot_lbfgs_convergence(lbfgs_result, minuit_result)

    return {"lbfgs": lbfgs_result, "minuit": minuit_result}


def plot_lbfgs_convergence(lbfgs_result, minuit_result):
    """Plot L-BFGS convergence with Minuit reference."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss vs iteration
    ax1.plot(lbfgs_result["x"], lbfgs_result["y"], "b-", linewidth=2, label="L-BFGS")
    ax1.axhline(y=minuit_result["fval"], color="r", linestyle="--", alpha=0.7, label=f"Minuit ({minuit_result['fval']:.4f})")
    ax1.set_xlabel("L-BFGS Step")
    ax1.set_ylabel("chi2")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Loss Convergence")

    # EDM vs iteration
    ax2.plot(lbfgs_result["x"], lbfgs_result["edm_values"], "b-", linewidth=2, label="L-BFGS EDM")
    ax2.axhline(y=minuit_result["edm"], color="r", linestyle="--", alpha=0.7, label=f"Minuit EDM ({minuit_result['edm']:.2e})")
    ax2.axhline(y=1e-5, color="gray", linestyle=":", alpha=0.5, label="Tolerance (1e-5)")
    ax2.set_xlabel("L-BFGS Step")
    ax2.set_ylabel("EDM")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title("EDM Convergence")

    plt.tight_layout()
    plt.savefig("lbfgs_vs_minuit.png", dpi=150, bbox_inches="tight")
    logger.info("Saved plot: lbfgs_vs_minuit.png")
    plt.show()


def demonstrate_edm_physics():
    """Demonstrate the physical meaning of EDM."""
    print("""
    EDM (Estimated Distance to Minimum):
    =====================================

    Definition:  EDM = 1/2 * g^T * H^(-1) * g

    Where g = gradient, H = Hessian matrix.

    Physical meaning: estimated function value difference from current
    point to the minimum under quadratic approximation.

    In L-BFGS, H^(-1) is approximated via stored curvature pairs
    (two-loop recursion), giving a natural EDM estimate without
    computing the full Hessian.

    Convergence thresholds:
      EDM < 1e-3 : coarse convergence
      EDM < 1e-4 : standard (iMinuit default)
      EDM < 1e-6 : high precision
    """)


if __name__ == "__main__":
    results = compare_lbfgs_vs_minuit()
    demonstrate_edm_physics()
