import math
from typing import Literal
from scipy.stats import norm 
from useful_classes import ConvergenceResult, EuropeanOption


def bs_price(option: EuropeanOption) -> float:
    
    # if the option has already reached maturity
    if option.T <= 0:
        if option.option_type == "call":
            return max(option.S0-option.K, 0.0)
        elif option.option_type == "put":
            return max(option.K-option.S0, 0.0)
        else:
            raise ValueError (f"Unknown option_type: {option.option_type}")
    # if the volatility is zero 
    if option.sigma <=0:
        forward = option.S0 * math.exp(option.r*option.T)
        if option.option_type == "call":
            payoff = max(forward-option.K, 0.0)
        elif option.option_type == "put":
            payoff = max(option.K-forward, 0.0)
        else:
            raise ValueError(f"Unknown option_type: {option.option_type}")

        return math.exp(-option.r * option.T) * payoff
    # otherwise, use the Black-Scholes closed form formula to evalute the final price of the option.
    d1 = (math.log(option.S0/option.K)+(option.r+0.5*option.sigma*option.sigma)*option.T)/(option.sigma *math.sqrt(option.T))
    d2 = d1 - option.sigma* math.sqrt(option.T)

    if option.option_type == "call":
        price = option.S0 * norm.cdf(d1) - option.K * math.exp(-option.r * option.T) * norm.cdf(d2)
    elif option.option_type == "put":
        price = option.K * math.exp(-option.r * option.T) * norm.cdf(-d2) - option.S0 * norm.cdf(-d1)
    else:
        raise ValueError(f"Unknown option_type: {option.option_type}")
    
    return float(price)