from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Tuple, Iterable, List, Literal, Optional

from bs_formula import bs_price
from mc_engine import mc_european_price
from variance_reduction import mc_european_price_antithetic
from useful_classes import ConvergenceResult, EuropeanOption

import numpy as np
from scipy.stats import norm 

def confidence_interval(
        mean: float,
        sample_std: float, 
        n_samples: int,
        alpha: float = 0.05,
) -> Tuple[float, float, float]:
    if n_samples <=1:
        raise ValueError ("n_samples must be greater than 1 for a confidence interval.")
    
    z = norm.ppf(1.0 - alpha/2.0)
    half_width = z * sample_std / sqrt(n_samples)
    lower = mean - half_width
    upper = mean + half_width
    return lower, upper, half_width


def convergence_study(
        option: EuropeanOption,
        n_paths_list:Iterable[int],
        use_antithetic: bool = False,
        alpha: float = 0.05,
        rng_seed: Optional[int] = 42,
) -> List[ConvergenceResult]:

    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed)
    else:
        rng = np.random.default_rng()

    true_price = bs_price(option)

    results: List[ConvergenceResult] = []

    for n_paths in n_paths_list:
        if use_antithetic:
            estimate, sample_std = mc_european_price_antithetic(option, n_paths = n_paths, rng = rng,)

        else:
            estimate, sample_std = mc_european_price(option, n_paths = n_paths, rng = rng,)

        ci_lower, ci_upper, half_width = confidence_interval(
            mean = estimate,
            sample_std = sample_std, 
            n_samples = n_paths,
            alpha = alpha,
        )

        abs_error = abs(estimate - true_price)

        results.append(
            ConvergenceResult(
                n_paths = n_paths,
                estimate = estimate,
                ci_lower = ci_lower,
                ci_upper = ci_upper,
                ci_half_width = half_width,
                abs_error_vs_bs = abs_error,
            )
        )

    return results

if __name__ == "__main__":
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
    option = EuropeanOption(S0, K, r, sigma, T)
    n_paths_list = [1_000, 5_000, 10_000, 50_000, 100_000]

    results_plain = convergence_study(
        option = option,
        n_paths_list=n_paths_list,
        use_antithetic=False,
        alpha=0.05,
        rng_seed=42,
    )

    for res in results_plain:
        print(
            f"N={res.n_paths:6d}  "
            f"estimate={res.estimate:.4f}  "
            f"CI=[{res.ci_lower:.4f}, {res.ci_upper:.4f}]  "
            f"half-width={res.ci_half_width:.4f}  "
            f"|error|={res.abs_error_vs_bs:.4f}"
        )
