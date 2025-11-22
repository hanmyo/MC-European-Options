from typing import Optional, Sequence, Tuple, Literal
import numpy as np
from useful_classes import ConvergenceResult, EuropeanOption

OptionType = Literal["call", "put"]

def compute_terminal_prices(
        option: EuropeanOption,
        Z: np.ndarray,
) -> np.ndarray:
    
    drift = (option.r-0.5 * option.sigma ** 2) * option.T 
    diffusion = option.sigma * np.sqrt(option.T) * Z 
    ST = option.S0 * np.exp(drift + diffusion)

    return ST    

def generate_terminal_prices(
        option: EuropeanOption,
        n_paths: int, 
        rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    
    if rng is None:
        rng = np.random.default_rng()
    
    if option.T < 0:
        raise ValueError("Time to maturity T must be non-negative.")
    
    if option.T == 0:
        return np.full(n_paths, S0, dtype="float")

    Z: np.ndarray = rng.standard_normal(n_paths)

    return compute_terminal_prices(option, Z)

def mc_european_price(
    option: EuropeanOption,
    n_paths: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")
    
    ST: np.ndarray = generate_terminal_prices(  option, n_paths = n_paths, rng = rng,)

    if option.option_type == "call":
        payoffs: np.ndarray = np.maximum(ST - option.K, 0.0)
    elif option.option_type == "put":
        payoffs: np.ndarray = np.maximum(option.K - ST, 0.0)
    else:
        raise ValueError("option_type must be either 'call' or 'put' ")
    
    discounted_payoffs: np.ndarray = np.exp(-option.r * option.T) * payoffs

    price_estimate: float = float(discounted_payoffs.mean())
    price_std: float = float(discounted_payoffs.std(ddof = 1))

    return price_estimate, price_std

# function to simulate for different n_s
def mc_european_price_for_ns(
        
        option: EuropeanOption,
        n_paths_list: Sequence[int],
        seed: int = 42,
        
) -> Sequence[Tuple[int, float, float]]:
    
    results: list[Tuple[int, float, float]] = []

    for i, N in enumerate(n_paths_list):
        if N<=0:
            raise ValueError("All n_paths values must be positive.")
        
        rng = np.random.default_rng(seed + i)
        price_estimate, price_std = mc_european_price(option, n_paths = N, rng = rng)
        results.append((N, price_estimate, price_std))

    return results

if __name__ == "__main__":
    from bs_formula import bs_price

    S0: float = 100.0
    K: float = 100.0
    r: float = 0.05
    sigma: float = 0.2
    T: float = 1.0
    n_paths: int = 100_0000
    rng = np.random.default_rng(123)
    option: EuropeanOption = EuropeanOption(S0, K, r, sigma, T)
    

    mc_price, mc_std = mc_european_price(option,n_paths = n_paths, rng = rng,)

    bs: float = bs_price(option)

    print(f"Black-Scholes price:  {bs:.4f}")
    print(f"MC estimate:          {mc_price:.4f}")
    print(f"MC std of payoffs:    {mc_std:.4f}")
    print(f"Absolute error:       {abs(mc_price - bs):.4f}")