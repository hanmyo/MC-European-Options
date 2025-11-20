from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Tuple, Iterable, List, Literal, Optional

from bs_formula import bs_price
from mc_engine import mc_european_price
from variance_reduction import mc_european_price_antithetic

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

@dataclass
class ConvergenceResult:
    n_paths: int
    estimate: float
    ci_lower: float
    ci_upper: float
    ci_half_width: float 
    abs_error_vs_bs: float 

def convergence_study(
        S0: float,
        K: float,
        r: float, 
        sigma: float, 
        T: float,
        option_type: Literal["call", "put"],
        n_paths_list:Iterable[int],
        use_antithetic: bool = False,
        alpha: float = 0.05,
        rng_seed: Optional[int] = 42,
) -> List[ConvergenceResult]:

    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed)
    else:
        rng = np.random.default_rng()

    true_price = bs_price(S0, K, r, sigma, T, option_type=option_type)

    results: List[ConvergenceResult] = []

    for n_paths in n_paths_list:
        if use_antithetic:
            estimate, sample_std = mc_european_price_antithetic(S0 = S0,
                                                                K = K,
                                                                r = r,
                                                                sigma = sigma,
                                                                T = T,
                                                                n_paths = n_paths,
                                                                option_type = option_type,
                                                                rng = rng,)

        else:
            estimate, sample_std = mc_european_price(S0 = S0,
                                                   K = K,
                                                   r = r,
                                                   sigma = sigma,
                                                   T = T,
                                                   n_paths = n_paths,
                                                   option_type = option_type,
                                                   rng = rng,)

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
    n_paths_list = [1_000, 5_000, 10_000, 50_000, 100_000]

    results_plain = convergence_study(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        option_type="call",
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
