from variance_reduction import mc_european_price_antithetic
from mc_engine import mc_european_price  # your plain MC
from bs_formula import bs_price
import numpy as np 
S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
n = 100_000

rng = np.random.default_rng(123)

mc_plain, std_plain = mc_european_price(S0, K, r, sigma, T, n_paths=n, option_type="call", rng=rng)
rng = np.random.default_rng(123)  # same seed for fair comparison
mc_anti, std_anti = mc_european_price_antithetic(S0, K, r, sigma, T, n_paths=n, option_type="call", rng=rng)

bs = bs_price(S0, K, r, sigma, T, "call")

print("BS closed-form:", bs)
print("MC plain      :", mc_plain, "  std =", std_plain)
print("MC antithetic :", mc_anti, "  std =", std_anti)

