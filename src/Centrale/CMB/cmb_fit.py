import numpy as np
import matplotlib.pyplot as plt
import time
import emcee
import corner
from tqdm import tqdm
import healpy as hp
from camb import CAMBparams, get_results

# ── Global settings ───────────────────────────────────────────────────────────
NSIDE = 256  # HEALPix resolution
NOISE_UK = 10.0  # white-noise per pixel [µK]
# lmax = 3×nside is the HEALPix-supported ceiling for TT (spin-0); map2alm iter=3 deconvolves
# the pixel window reliably in this range. Goes to ell~768: covers peaks 1, 2, and base of 3.
LMAX_FACTOR = 3  # multiply by nside to get lmax

# Planck-inspired linear binning (Plik likelihood uses uniform Δell, not log-spaced).
# Log bins under-sample the acoustic peaks; uniform bins give ~equal modes per bin
# and put several bins across each peak regardless of its ell position.
DELTA_ELL_LO = 5  # bin width for ell < ELL_BREAK (feature-rich Sachs-Wolfe regime)
DELTA_ELL_HI = 30  # bin width for ell ≥ ELL_BREAK (acoustic peaks regime, Planck uses ~25-35)
ELL_BREAK = 50  # transition multipole between fine and coarse binning


# ── Helpers ───────────────────────────────────────────────────────────────────
def camb_spectrum(params, lmax):
    """Return (ell, C_ell) [µK²] from CAMB.

    Free parameters: omch2, ombh2, H0, As.
    ns and tau are left at CAMB defaults (ns=0.96, tau~0.054).
    """
    p = CAMBparams()
    p.set_cosmology(H0=params["H0"], ombh2=params["ombh2"], omch2=params["omch2"])
    p.InitPower.set_params(ns=0.96, As=params.get("As", 2.1e-9))
    p.set_for_lmax(lmax, lens_potential_accuracy=0)
    powers = get_results(p).get_cmb_power_spectra(p, CMB_unit="muK", raw_cl=True)
    cl = powers["total"][: lmax + 1, 0]
    return np.arange(lmax + 1, dtype=float), cl


def bin_spectrum(ell, cl, bin_edges):
    """
    (2ℓ+1)-weighted bandpower average using vectorised np.digitize + np.bincount.
    Returns (ell_centres, cl_binned, n_modes_per_bin) for non-empty bins only.
    """
    ell, cl = np.asarray(ell, dtype=float), np.asarray(cl, dtype=float)
    mask = (ell > 0) & (ell >= bin_edges[0]) & (ell < bin_edges[-1])
    el, cl_ = ell[mask], cl[mask]
    w   = 2 * el + 1
    idx = np.digitize(el, bin_edges) - 1   # 0-indexed bin for each mode
    n   = len(bin_edges) - 1

    W   = np.bincount(idx, weights=w,        minlength=n)
    Wel = np.bincount(idx, weights=w * el,   minlength=n)
    Wcl = np.bincount(idx, weights=w * cl_,  minlength=n)

    valid = W > 0
    return Wel[valid] / W[valid], Wcl[valid] / W[valid], W[valid]


def to_dl(ell, cl):
    """D_ell = ell(ell+1)/(2π) · C_ell."""
    ell = np.asarray(ell, dtype=float)
    return np.where(ell > 0, ell * (ell + 1) / (2 * np.pi) * cl, 0.0)


# ── Analyser ──────────────────────────────────────────────────────────────────
class CMBAnalyzer:
    def __init__(self, nside=NSIDE, noise_uk=NOISE_UK):
        self.nside = nside
        self.lmax = LMAX_FACTOR * nside
        self.npix = 12 * nside**2
        self.N_ell    = noise_uk**2 * 4 * np.pi / self.npix
        self.noise_uk = noise_uk

        # Planck-inspired hybrid linear binning:
        #   fine bins (Δell=DELTA_ELL_LO) at low ell to resolve the Sachs-Wolfe plateau
        #   coarse bins (Δell=DELTA_ELL_HI) across the acoustic peaks
        lo = np.arange(10, ELL_BREAK, DELTA_ELL_LO)
        hi = np.arange(ELL_BREAK, self.lmax + DELTA_ELL_HI, DELTA_ELL_HI)
        self.bin_edges = np.unique(np.concatenate([lo, hi]))

    def observe(self, params_true):
        """
        Synthesise a CMB sky + noise, measure the TT spectrum, return binned bandpowers.
        Also returns the unbinned (ell, cl_obs) for plotting.
        """
        _, cl_true = camb_spectrum(params_true, self.lmax)
        sky  = hp.alm2map(hp.synalm(cl_true, lmax=self.lmax), nside=self.nside)
        sky += np.random.normal(0, self.noise_uk, self.npix)

        cl_obs = hp.alm2cl(hp.map2alm(sky, lmax=self.lmax))
        ell_obs = np.arange(len(cl_obs), dtype=float)
        ell_b, cl_b, modes_b = bin_spectrum(ell_obs, cl_obs, self.bin_edges)
        return ell_b, cl_b, modes_b, ell_obs, cl_obs

    def predict(self, params):
        """
        Binned theory bandpowers, directly comparable to the observed bandpowers.
        """
        ell, cl = camb_spectrum(params, self.lmax)
        return bin_spectrum(ell, cl, self.bin_edges)

    def fit(self, cl_b_obs, modes_b, n_walkers=16, n_steps=300, n_burn=50):
        """
        Sample the posterior with emcee.

        Returns (best_params, sampler, flat_samples, chi2_at_median).
        best_params = posterior medians.
        """

        param_keys = ["omch2", "ombh2", "H0", "As"]
        bound_lo = np.array([0.05, 0.010, 50.0, 1.5e-9])
        bound_hi = np.array([0.30, 0.050, 90.0, 3.5e-9])

        def log_prior(p):
            if np.any(p < bound_lo) or np.any(p > bound_hi):
                return -np.inf
            return 0.0  # flat prior within bounds

        def log_likelihood(p):
            try:
                _, cl_b_th, _ = self.predict({k: v for k, v in zip(param_keys, p)})
                # Knox variance: Var(C_b) = 2/n_modes × (C_b + N_ell)²
                # C_b term = cosmic/sample variance; N_ell = noise contribution.
                var = 2 / modes_b * (cl_b_th + self.N_ell) ** 2
                return -0.5 * float(np.sum((cl_b_obs - cl_b_th) ** 2 / var))
            except Exception:
                return -np.inf

        def log_posterior(p):
            lp = log_prior(p)
            return lp + log_likelihood(p) if np.isfinite(lp) else -np.inf

        # Initialise walkers as a tight ball around Planck-like fiducial values.
        rng = np.random.default_rng(42)
        p0_center = np.array([0.12, 0.022, 67.3, 2.1e-9])
        p0_sigma = np.array([0.01, 0.002, 2.0, 0.1e-9])
        p0 = p0_center + p0_sigma * rng.standard_normal((n_walkers, 4))

        sampler = emcee.EnsembleSampler(n_walkers, 4, log_posterior)

        with tqdm(total=n_steps, desc="  MCMC", unit="step", ncols=72) as pbar:
            for _ in sampler.sample(p0, iterations=n_steps, progress=False):
                pbar.update(1)
                pbar.set_postfix(acc=f"{np.mean(sampler.acceptance_fraction):.2f}")

        flat_samples = sampler.get_chain(discard=n_burn, flat=True)
        medians = np.median(flat_samples, axis=0)
        best_params = {k: v for k, v in zip(param_keys, medians)}

        # Chi2 at posterior median
        _, cl_b_th, _ = self.predict(best_params)
        var = 2 / modes_b * (cl_b_th + self.N_ell) ** 2
        chi2_med = float(np.sum((cl_b_obs - cl_b_th) ** 2 / var))

        return best_params, sampler, flat_samples, chi2_med

    # ── Diagnostic plots ─────────────────────────────────────────────────────
    def plot_walkers(self, sampler, params_true, n_burn):
        """Plot MCMC walker chains for each parameter."""
        chain = sampler.get_chain()  # (n_chain_steps, n_walkers, n_params)
        # As displayed as ln(10^10 As) — the standard cosmology convention (~3.04)
        chain_As_log = np.log(chain[:, :, 3] * 1e10)
        labels = [r"$\Omega_c h^2$", r"$\Omega_b h^2$", r"$H_0$", r"$\ln(10^{10}A_s)$"]
        truths = [
            params_true["omch2"],
            params_true["ombh2"],
            params_true["H0"],
            np.log(params_true["As"] * 1e10),
        ]
        n_chain_steps, n_walkers, _ = chain.shape

        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        for i, (ax, label, truth) in enumerate(zip(axes, labels, truths)):
            data = chain_As_log if i == 3 else chain[:, :, i]
            ax.plot(data, "k-", alpha=0.08, linewidth=0.7)
            ax.plot(
                np.median(data, axis=1),
                "tab:blue",
                linewidth=1.5,
                label="Median" if i == 0 else "",
            )
            ax.axhline(
                truth,
                color="tab:red",
                linestyle="--",
                linewidth=1.5,
                label="True" if i == 0 else "",
            )
            ax.axvspan(0, n_burn, alpha=0.12, color="gray", label="Burn-in" if i == 0 else "")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)

        axes[-1].set_xlabel("Step")
        axes[0].legend(loc="upper right", fontsize=9)
        acc = np.mean(sampler.acceptance_fraction)
        fig.suptitle(f"MCMC Walker Chains  (acceptance rate = {acc:.2f})", fontsize=13)
        plt.tight_layout()
        plt.savefig("cmb_walkers.png", dpi=150, bbox_inches="tight")
        print("\nSaved walker plot to cmb_walkers.png")
        return fig

    def plot_corner(self, flat_samples, params_true):
        """Triangle (corner) plot of the marginalised posterior."""
        # Transform As → ln(10^10 As) for readable axis values
        samples_plot = flat_samples.copy()
        samples_plot[:, 3] = np.log(samples_plot[:, 3] * 1e10)
        labels = [r"$\Omega_c h^2$", r"$\Omega_b h^2$", r"$H_0$", r"$\ln(10^{10}A_s)$"]
        truths = [
            params_true["omch2"],
            params_true["ombh2"],
            params_true["H0"],
            np.log(params_true["As"] * 1e10),
        ]

        fig = corner.corner(
            samples_plot,
            labels=labels,
            truths=truths,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 9},
            truth_color="tab:red",
            smooth=1.0,
            smooth1d=1.0,
        )
        fig.suptitle("Marginalised posterior (MCMC)", fontsize=13, y=1.01)
        plt.savefig("cmb_corner.png", dpi=150, bbox_inches="tight")
        print("\n✓ Saved corner plot to cmb_corner.png")
        return fig

    def plot(self, ell_obs, cl_obs, ell_b_obs, cl_b_obs, params_true, params_fit, chi2_fit):
        """Spectrum comparison (top) + per-parameter recovery (bottom)."""
        _, cl_true = camb_spectrum(params_true, self.lmax)
        _, cl_fit = camb_spectrum(params_fit, self.lmax)
        ell_t = np.arange(self.lmax + 1, dtype=float)

        fig = plt.figure(figsize=(14, 9))
        gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.55)

        # ── Power spectrum ────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, :])
        ax.plot(
            ell_obs[2:],
            to_dl(ell_obs[2:], cl_obs[2:]),
            "k-",
            alpha=0.15,
            linewidth=0.7,
            label="Unbinned (noisy)",
        )
        ax.plot(ell_t[2:], to_dl(ell_t[2:], cl_true[2:]), "b-", linewidth=2, label="Theory (true)")
        ax.plot(
            ell_t[2:], to_dl(ell_t[2:], cl_fit[2:]), "r--", linewidth=1.8, label="Theory (fitted)"
        )
        ax.errorbar(
            ell_b_obs,
            to_dl(ell_b_obs, cl_b_obs),
            yerr=to_dl(ell_b_obs, cl_b_obs) * np.sqrt(2 / (2 * ell_b_obs + 1)),
            fmt="ko",
            markersize=5,
            zorder=5,
            label="Binned bandpowers",
        )
        ax.axvline(
            self.lmax,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label=f"lmax = {self.lmax}",
        )
        ax.set_xlabel(r"Multipole $\ell$")
        ax.set_ylabel(r"$D_\ell^{TT}$ [µK²]")
        ax.set_title(f"CMB TT Power Spectrum  (χ² = {chi2_fit:.1f})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # ── Parameter recovery ────────────────────────────────────────────────
        for col, (key, label) in enumerate(
            zip(
                ["omch2", "ombh2", "H0", "As"],
                [r"$\Omega_c h^2$", r"$\Omega_b h^2$", r"$H_0$", r"$A_s\ [10^{-9}]$"],
            )
        ):
            ax = fig.add_subplot(gs[1, col])
            # Display As in units of 10^-9 for readability
            scale = 1e9 if key == "As" else 1.0
            t = params_true[key] * scale
            f = params_fit[key] * scale
            err = 100 * (f - t) / t
            clr = "tab:green" if abs(err) < 5 else "tab:orange" if abs(err) < 20 else "tab:red"
            ax.bar(["True", "Fitted"], [t, f], color=["steelblue", clr], alpha=0.85, width=0.5)
            ax.set_title(label, fontsize=11)
            ax.grid(alpha=0.3, axis="y")
            sign = "+" if err >= 0 else ""
            ax.text(
                0.5,
                0.97,
                f"{sign}{err:.1f}%",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=10,
                color=clr,
                fontweight="bold",
            )
            for xi, val in enumerate([t, f]):
                ax.text(xi, val, f"{val:.4f}", ha="center", va="bottom", fontsize=8)

        plt.savefig("cmb_analysis.png", dpi=150, bbox_inches="tight")
        print("\n✓ Saved plot to cmb_analysis.png")
        return fig


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("CMB Power Spectrum Analysis")
    print("=" * 60)

    params_true = {"omch2": 0.120, "ombh2": 0.0224, "H0": 67.3, "As": 2.1e-9}
    analyzer = CMBAnalyzer(nside=NSIDE, noise_uk=NOISE_UK)

    n_bins_eff = len(analyzer.bin_edges) - 1
    print(
        f"\n[1] Generating CMB map + measuring binned spectrum "
        f"(nside={NSIDE}, lmax={analyzer.lmax}, ~{n_bins_eff} bins)..."
    )
    t0 = time.time()
    ell_b, cl_b, modes_b, ell_obs, cl_obs = analyzer.observe(params_true)
    print(f"    ✓ Done in {time.time() - t0:.2f}s  ({len(ell_b)} bands)")

    N_WALKERS, N_STEPS, N_BURN = 16, 300, 50
    print(f"\n[2] MCMC sampling ({N_WALKERS} walkers × {N_STEPS} steps, {N_BURN} burn-in)...")
    t0 = time.time()
    params_fit, sampler, flat_samples, chi2_fit = analyzer.fit(
        cl_b, modes_b, n_walkers=N_WALKERS, n_steps=N_STEPS, n_burn=N_BURN
    )
    print(f"    ✓ Done in {time.time() - t0:.1f}s  χ² = {chi2_fit:.2f}")

    print("\n" + "=" * 60 + "\nRESULTS (posterior medians)\n" + "=" * 60)
    print(f"\n{'Parameter':<15} {'True':<12} {'Fitted':<12} {'Error %'}")
    print("-" * 55)
    for key, name, scale in [
        ("omch2", "Ωc h²", 1),
        ("ombh2", "Ωb h²", 1),
        ("H0", "H₀", 1),
        ("As", "As [1e-9]", 1e9),
    ]:
        t, f = params_true[key] * scale, params_fit[key] * scale
        print(f"{name:<15} {t:<12.5f} {f:<12.5f} {100 * abs(f - t) / t:.2f}%")
    print("\n" + "=" * 60)

    print("\n[3] Plotting results...")
    analyzer.plot(ell_obs, cl_obs, ell_b, cl_b, params_true, params_fit, chi2_fit)
    analyzer.plot_walkers(sampler, params_true, n_burn=N_BURN)
    analyzer.plot_corner(flat_samples, params_true)
    print("Done!")


if __name__ == "__main__":
    main()
