from typing import Literal, Optional, Tuple
import numpy as np
from mc_engine import compute_terminal_prices
from useful_classes import EuropeanOption

def mc_european_price_antithetic(
        option: EuropeanOption,
        n_paths: int, 
        rng: Optional[np.random.Generator] = None, 
) -> Tuple[float, float]:

    if rng is None:
        rng = np.random.default_rng() 

    if n_paths % 2 != 0:
        raise ValueError("n_paths must be even for antithetic variates.")
    
    m = n_paths // 2
    Z: np.ndarray = rng.standard_normal(m)
    Z_pair: np.ndarray = np.concatenate([Z, -Z])

    ST: np.ndarray = compute_terminal_prices(option, Z_pair)

    if option.option_type == "call":
        payoffs = np.maximum(ST-option.K, 0.0)

    elif option.option_type == "put":
        payoffs = np.maximum(option.K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call or 'put' ")

    discount_factor = np.exp(-option.r * option.T)
    discounted_payoffs = discount_factor * payoffs

    pair_means = 0.5 * (
        discounted_payoffs[:m] + discounted_payoffs[m:]
    )

    price_estimate = float(pair_means.mean())
    price_std = float(pair_means.std(ddof=1))

    return price_estimate, price_std