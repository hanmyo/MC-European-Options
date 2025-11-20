import math

from typing import Literal

from scipy.stats import norm 

OptionType = Literal["call", "put"]

def bs_price(
        S0:float,
        K: float, 
        r:float,
        sigma: float,
        T: float, 
        option_type: OptionType="call",
) -> float:
    
    if T <= 0:
        if option_type == "call":
            return max(S0-K, 0.0)
        elif option_type == "put":
            return max(K-S0, 0.0)
        else:
            raise ValueError (f"Unknown option_type: {option_type}")

    if sigma <=0:
        forward = S0 * math.exp(r*T)
        if option_type == "call":
            payoff = max(forward-K, 0.0)
        elif option_type == "put":
            payoff = max(K-forward, 0.0)
        else:
            raise ValueError(f"Unknown option_type: {option_type}")

        return math.exp(-r * T) * payoff
    
    d1 = (math.log(S0/K)+(r+0.5*sigma*sigma)*T)/(sigma *math.sqrt(T))
    d2 = d1 - sigma* math.sqrt(T)

    if option_type == "call":
        price = S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError(f"Unknown option_type: {option_type}")
    
    return float(price)