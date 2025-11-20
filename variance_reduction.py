from typing import Literal, Optional, Tuple
import numpy as np

OptionType = Literal["call", "put"]

def mc_european_price_antithetic(
        S0: float, 
        K: float,
        r: float, 
        sigma: float, 
        T: float, 
        n_paths: int, 
        option_type : OptionType = "call",
        rng: Optional[np.random.Generator] = None, 
) -> Tuple[float, float]:

    if rng is None:
        rng = np.random.default_rng() 

    if n_paths % 2 != 0:
        raise ValueError("n_paths must be even for antithetic variates.")
    
    m = n_paths // 2
    Z = rng.standard_normal(m)
    Z_pair = np.concatenate([Z, -Z])

    drift = (r - 0.5 * sigma ** 2) * T
    diffusion_scale = sigma * np.sqrt(T)

    ST = S0 * np.exp(drift + diffusion_scale * Z_pair)

    if option_type == "call":
        payoffs = np.maximum(ST-K, 0.0)

    elif option_type == "put":
        payoffs = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call or 'put' ")

    discount_factor = np.exp(-r * T)
    discounted_payoffs = discount_factor * payoffs

    pair_means = 0.5 * (
        discounted_payoffs[:m] + discounted_payoffs[m:]
    )

    price_estimate = float(pair_means.mean())
    price_std = float(pair_means.std(ddof=1))

    return price_estimate, price_std