import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on sys.path for `from src...` imports when run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import compare_strategies
from src.utils import sharpe, annualize_return, annualize_vol, max_drawdown


def ensure_outputs_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_equity_curves(results: dict, outdir: Path):
    for key, res in results.items():
        ec = res["equity_curve"]
        plt.figure(figsize=(10, 5))
        ec.plot()
        plt.title(f"Equity Curve — {key}")
        plt.ylabel("Growth of 1")
        plt.xlabel("Date")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"equity_curve_{key}.png", dpi=150)
        plt.close()


def save_drawdowns(results: dict, outdir: Path):
    for key, res in results.items():
        ec = res["equity_curve"]
        dd = ec / ec.cummax() - 1.0
        plt.figure(figsize=(10, 3))
        dd.plot(color="crimson")
        plt.title(f"Drawdowns — {key}")
        plt.ylabel("Drawdown")
        plt.xlabel("Date")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"drawdowns_{key}.png", dpi=150)
        plt.close()


def save_weights_heatmap(results: dict, outdir: Path, top_n: int = 15):
    sns.set_style("whitegrid")
    for key, res in results.items():
        W = res.get("weights")
        if W is None or W.empty:
            continue
        # Keep top assets by average absolute weight
        avg_abs = W.abs().mean().sort_values(ascending=False)
        keep = avg_abs.head(top_n).index
        W_small = W[keep]
        plt.figure(figsize=(12, 6))
        sns.heatmap(W_small.T, cmap="viridis", cbar=True)
        plt.title(f"Weights Heatmap (top {top_n}) — {key}")
        plt.xlabel("Date")
        plt.ylabel("Ticker")
        plt.tight_layout()
        plt.savefig(outdir / f"weights_heatmap_{key}.png", dpi=150)
        plt.close()


def write_summary_csv(results: dict, outdir: Path):
    rows = []
    for key, res in results.items():
        r = res["net"]
        ec = res["equity_curve"]
        rows.append(dict(
            strategy=key,
            sharpe=sharpe(r),
            ann_return=annualize_return(r),
            ann_vol=annualize_vol(r),
            max_drawdown=max_drawdown(ec),
            turnover=float(res.get("turnover", pd.Series(dtype=float)).mean() if res.get("turnover") is not None else np.nan),
        ))
    pd.DataFrame(rows).to_csv(outdir / "summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Run RMT backtests and save plots to outputs/")
    parser.add_argument("--returns_csv", default="data/processed/returns_log.csv")
    parser.add_argument("--window", type=int, default=504)
    parser.add_argument("--rebalance", type=int, default=21)
    parser.add_argument("--methods", nargs="+", default=["gmv", "riskparity"], choices=["gmv", "riskparity"]) 
    parser.add_argument("--tc_bps", type=float, default=1.0)
    parser.add_argument("--use_cleaned", choices=["raw", "cleaned", "both"], default="both")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    returns = pd.read_csv(args.returns_csv, index_col=0, parse_dates=True)

    if args.use_cleaned == "both":
        results = {}
        for m in args.methods:
            # raw
            results[f"{m}_raw"] = compare_strategies(returns, args.window, args.rebalance, methods=(m,), tc_bps=args.tc_bps)[f"{m}_raw"]
            # cleaned
            results[f"{m}_cleaned"] = compare_strategies(returns, args.window, args.rebalance, methods=(m,), tc_bps=args.tc_bps)[f"{m}_cleaned"]
    else:
        tag = "cleaned" if args.use_cleaned == "cleaned" else "raw"
        results = {}
        tmp = compare_strategies(returns, args.window, args.rebalance, methods=tuple(args.methods), tc_bps=args.tc_bps)
        for m in args.methods:
            results[f"{m}_" + tag] = tmp[f"{m}_" + tag]

    outdir = ensure_outputs_dir(Path(args.outdir))
    save_equity_curves(results, outdir)
    save_drawdowns(results, outdir)
    save_weights_heatmap(results, outdir)
    write_summary_csv(results, outdir)
    print(f"Saved outputs to {outdir.resolve()}")


if __name__ == "__main__":
    main()


