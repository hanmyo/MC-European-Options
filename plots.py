from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt

from diagnostics import ConvergenceResult

def plot_convergence(
        results: Sequence[ConvergenceResult],
        bs_ref: float,
        plot_prefix: str,
        option_type: str,
        use_antithetic: bool,
) -> None:
    
    sorted_results = sorted(results, key = lambda r: r.n_paths)

    n_paths = [r.n_paths for r in sorted_results]
    estimates = [r.estimate for r in sorted_results]
    ci_lower = [r.ci_lower for r in sorted_results]
    ci_upper = [r.ci_upper for r in sorted_results]
    abs_errors = [r.abs_error_vs_bs for r in sorted_results]

    method_label = "MC + Antithetic" if use_antithetic else "Plain MC"

    prefix_path = Path(plot_prefix)
    if prefix_path.suffix:
        out_dir = prefix_path.parent
    else:
        out_dir = prefix_path 
    out_dir.mkdir(parents = True, exist_ok = True)

    plt.figure()

    plt.plot(n_paths, estimates, marker = "o", linestyle ="-", label = f"MC estimate ({method_label})")
    plt.axhline(bs_ref, linestyle = "--", label="Black_scholes price")
    
    plt.xscale("log")   
    plt.xlabel("Number of Monte Carlo paths (log scale)")
    plt.ylabel("Option price")
    plt.title(f"Convergence of {option_type} price ({method_label})")
    plt.legend()
    plt.grid(True, which = "both", linestyle = ":")

    price_plot_path = out_dir / f"{prefix_path.stem}_price_convergence.png"
    plt.tight_layout()
    plt.savefig(price_plot_path, dpi = 200)
    plt.close()

    print(f"Saved price convergence plot to {price_plot_path}")

    plt.figure()

    plt.plot(n_paths, abs_errors, marker = "o", linestyle = "-", label = "|MC - BS|")

    plt.xscale("log")
    plt.yscale("log") #error decays like ~1/sqrt(N)

    plt.xlabel("Number of Monte Carlo paths (log scale)")
    plt.ylabel("Absolute error vs Black-Scholes (log scale)")
    plt.title(f"Convergence of error ({method_label})")
    plt.legend()
    plt.grid(True, which="both", linestyle = ":")

    error_plot_path = out_dir / f"{prefix_path.stem}_error_convergence.png"
    plt.tight_layout()
    plt.savefig(error_plot_path, dpi = 200)
    plt.close()

    print(f"Saved error convergence plot to {error_plot_path}")


