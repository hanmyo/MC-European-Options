from typing import Literal 
from dataclasses import dataclass
OptionType = Literal["call", "put"]

@dataclass 
class EuropeanOption:
    S0:float        #Spot Price
    K: float        #Strike Price
    r: float        #Risk-free interest rate
    sigma: float    #Volatility
    T: float        #Time to maturity
    option_type: OptionType = "call" #default type

@dataclass
class ConvergenceResult:
    n_paths: int
    estimate: float
    ci_lower: float
    ci_upper: float
    ci_half_width: float 
    abs_error_vs_bs: float 