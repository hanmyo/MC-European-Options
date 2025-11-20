from typing import Optional, Sequence, Tuple, Literal

import numpy as np

OptionType = Literal["call", "put"]

def generate_terminal_prices(
        S0: float,
        r: float,
        sigma: float,
        T: float, 
        n_paths: int, 
        rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    
    if rng is None:
        rng = np.random.default_rng()
    
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative.")
    
    if T == 0:
        return np.full(n_paths, S0, dtype="float")
    
    Z = rng.standard_normal(n_paths)

    drift = (r-0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * Z

    ST = S0 * np.exp(drift+diffusion)

    return ST

def mc_european_price(
        
    S0: float, 
    K: float, 
    r: float, 
    sigma: float,
    T: float, 
    n_paths: int,
    option_type: OptionType = "call",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")
    
    ST: np.ndarray = generate_terminal_prices(
        S0 = S0,
        r = r, 
        sigma = sigma,
        T = T, 
        n_paths = n_paths, 
        rng = rng,
    )

    if option_type == "call":
        payoffs: np.ndarray = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        payoffs: np.ndarray = np.maximum(K-ST, 0.0)
    else:
        raise ValueRoor("option_type must be either 'call' or 'put' ")
    
    discounted_payoffs: np.ndarray = np.exp(-r * T) * payoffs

    price_estimate: float = float(discounted_payoffs.mean())
    price_std: float = float(discounted_payoffs.std(ddof = 1))

    return price_estimate, price_std


def mc_european_price_for_ns(
        S0: float,
        K: float, 
        r: float,
        sigma: float, 
        T: float,
        n_paths_list: Sequence[int],
        option_type: OptionType = "call",
        seed: int = 42,
) -> Sequence[Tuple[int, float, float]]:
    
    results: list[Tuple[int, float, float]] = []

    for i, N in enumerate(n_paths_list):
        if N<=0:
            raise ValueError("All n_paths values must be positive.")
        
        rng = np.random.default_rng(seed + i)
        price_estimate, price_std = mc_european_price(
            S0 = S0,
            K = K,
            r = r, 
            sigma = sigma, 
            T = T, 
            n_paths = N,
            option_type = option_type,
            rng = rng,
        )
        results.append((N, price_estimate, price_std))

    return results

if __name__ == "__main__":
    from bs_formula import bs_price

    S0: float = 100.0
    K: float = 100.0
    r: float = 0.05
    sigma: float = 0.2
    T: float = 1.0
    n_paths: int = 100_000

    rng = np.random.default_rng(123)

    mc_price, mc_std = mc_european_price(
        S0 = S0,
        K = K, 
        r = r, 
        sigma = sigma, 
        T = T, 
        n_paths = n_paths,
        option_type = "call",
        rng = rng,
    )

    bs: float = bs_price(S0, K, r, sigma, T, option_type= "call")

    print(f"Black-Scholes price:  {bs:.4f}")
    print(f"MC estimate:          {mc_price:.4f}")
    print(f"MC std of payoffs:    {mc_std:.4f}")
    print(f"Absolute error:       {abs(mc_price - bs):.4f}")