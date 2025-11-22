from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence
from diagnostics import convergence_study
from useful_classes import EuropeanOption, ConvergenceResult
from bs_formula import bs_price 


def parse_args () -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Monte Carlo pricing and convergence diagnostics for European Options."
    )
    # model parameters
    parser.add_argument("--S0", type=float, required = True, help = "Initial spot price")
    parser.add_argument("--K", type = float, required = True, help = "Strike price")
    parser.add_argument("--r", type=float, required = True, help = "Risk-free interest rate (annual, continuously compounded).")
    parser.add_argument("--sigma", type = float, required = True, help = "Volatility (annual)")
    parser.add_argument("--T", type = float, required = True, help = "Time to maturity")

    parser.add_argument(
        "--option-type",
        choices = ["call", "put"],
        default = "call",
        help = "Option type: call or put (default: call).",
    )

    # settings for Monte Carlo
    parser.add_argument(
        "--n-paths-list",
        type = int,
        nargs = "+",
        required = True,
        help = "List of MOnte Carlo path conuts for convergence (e.g. 1000 5000 10000)."
    )

    parser.add_argument(
        "--antithetic",
        action = "store_true",
        help = "Use antithetic variates for variance reduction.",
    )

    parser.add_argument(
        "--alpha",
        type = float,
        default = 0.05,

        help = "Significance level for (1 - alpha) confidence intervals (default: 0.05 for 95% CIs)"
    )
    parser.add_argument(
        "--seed",
        type = int,
        default = 42, 
        help = "random seed for reproducibility (default: 42)."
    )

    # Output 
    parser.add_argument(
        "--csv-path",
        type = str,
        default = None,
        help = "Optional path to save results as CSV."
    )

    parser.add_argument(
        "--plot-prefix",
        type = str,
        default = None,
        help = "Optional prefix for saving convergence plots (e.g. plots/call_)."
    )
    return parser.parse_args()

def print_table(results: Sequence[ConvergenceResult], bs_ref: float, use_antithetic: bool) -> None:

    method = "MC + Antithetic" if use_antithetic else "Plain MC"
    print(f"\nConvergence results ({method})")
    print(f"Black-Scholes reference price: {bs_ref:.6f}\n")

    """  n_paths: int
    estimate: float
    ci_lower: float
    ci_upper: float
    ci_half_width: float 
    abs_error_vs_bs: float 
    """
    header = (
        f"{'n_paths':>10}"
        f"{'estimate':>12}"
        f"{'ci_lower':>12}"
        f"{'ci_upper':>12}"
        f"{'ci_half_width':>12}"
        f"{'abs_error_vs_bs':>12}"
    )

    print(header)
    print("-" * len(header))

    for res in results:
        print(
            f"{res.n_paths:10d}"
            f"{res.estimate:12.6f}"
            f"{res.ci_lower:12.6f}"
            f"{res.ci_upper:12.6f}"
            f"{res.ci_half_width:12.6f}"
            f"{res.abs_error_vs_bs:12.6f}"
        )

    print()


def save_results_to_csv(
        results: Sequence[ConvergenceResult],
        csv_path: str,
        bs_ref: float,
) -> None:
    
    path = Path(csv_path)
    path.parent.mkdir(parents= True, exist_ok = True)

    fieldnames = [
        "n_paths",
        "estimate",
        "ci_lower",
        "ci_upper",
        "ci_half_width",
        "abs_error_vs_bs",
        "bs_ref",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()

        for res in results:
            writer.writerow(
                {
                    "n_paths": res.n_paths,
                    "estimate": res.estimate,
                    "ci_lower": res.ci_lower,
                    "ci_upper": res.ci_upper,
                    "ci_half_width": res.ci_half_width,
                    "abs_error_vs_bs": res.abs_error_vs_bs,
                    "bs_ref": bs_ref,    
               }
            )

    print(f"Saved results to {path}")

def main() -> None: 
    args = parse_args()
    option: EuropeanOption = EuropeanOption(args.S0, args.K, args.r, args.sigma, args.T, args.option_type)
    bs_ref = bs_price(option)

    results = convergence_study(
        option = option,
        n_paths_list = args.n_paths_list,
        use_antithetic = args.antithetic,
        alpha = args.alpha,
        rng_seed = args.seed,
    )

    print_table(results, bs_ref, use_antithetic = args.antithetic)

    if args.csv_path is not None:
        save_results_to_csv(results, args.csv_path, bs_ref)

    if args.plot_prefix is not None: 
        try:
            from plots import plot_convergence
            plot_convergence(
                results = results,
                bs_ref = bs_ref,
                plot_prefix = args.plot_prefix, 
                option_type = args.option_type,
                use_antithetic = args.antithetic,
            )
        except ImportError:
            print("Warning: plots module not found.")


if __name__ == "__main__":
    main()