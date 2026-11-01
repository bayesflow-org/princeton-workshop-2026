"""PyMC model construction and posterior subsampling for the HNN notebook.

The eight priors match helpers.PRIOR_SPECS and the augmented training data.
The neural likelihood receives log parameters in PARAMETER_NAMES order.
"""

from math import log

import numpy as np
import pymc as pm

from .helpers import PARAMETER_NAMES, PRIOR_SPECS


def sample_nle_posterior(
    neural_likelihood,
    observed_summary,
    *,
    draws=1000,
    tune=1000,
    chains=4,
    sampler="nuts",
    random_seed=2026,
    progressbar=True,
):
    """Sample physical parameters with the trained BayesFlow likelihood.

    Each call runs a new MCMC analysis. Sequential chains (cores=1) also work
    when this function is called from a notebook on Windows.
    """
    if sampler not in {"nuts", "smc"}:
        raise ValueError("sampler must be 'nuts' or 'smc'")

    def make_prior(name):
        spec = PRIOR_SPECS[name]
        if spec["family"] == "lognormal":
            return pm.LogNormal(name, mu=log(spec["median"]), sigma=spec["log_sd"])
        if spec["family"] == "gamma":
            return pm.Gamma(name, alpha=spec["shape"], beta=1.0 / spec["scale"])
        excess = pm.LogNormal(
            "measurement_noise_excess",
            mu=log(spec["median"]),
            sigma=spec["log_sd"],
        )
        return pm.Deterministic(name, spec["offset"] + excess)

    with pm.Model():
        parameters = {name: make_prior(name) for name in PARAMETER_NAMES}
        pm.Deterministic("feedback_time", parameters["early_feedforward_time"] + parameters["feedback_delay"])
        pm.Deterministic(
            "late_feedforward_time",
            parameters["early_feedforward_time"]
            + parameters["feedback_delay"]
            + parameters["late_feedforward_delay"],
        )
        likelihood_parameters = {
            f"log_{name}": pm.math.log(value)
            for name, value in parameters.items()
            if name != "measurement_noise_sd"
        }
        likelihood_parameters["log_measurement_noise_excess"] = pm.math.log(
            parameters["measurement_noise_sd"] - PRIOR_SPECS["measurement_noise_sd"]["offset"]
        )
        neural_likelihood(
            "dipole_summary",
            observed=observed_summary.ravel(),
            **likelihood_parameters,
        )

        if sampler == "smc":
            return pm.sample_smc(
                draws=draws, chains=chains, cores=1, random_seed=random_seed, progressbar=progressbar
            )
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            target_accept=0.9,
            random_seed=random_seed,
            progressbar=progressbar,
        )

def subset_draws(draws, n_draws, *, rng):
    """Select posterior rows with a caller-owned NumPy random generator."""
    indices = rng.choice(len(next(iter(draws.values()))), size=n_draws, replace=False)
    return {name: np.asarray(draws[name])[indices, None] for name in PARAMETER_NAMES}
