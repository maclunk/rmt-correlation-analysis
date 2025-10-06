# Random Matrix Theory for Equity Correlations

This project implements a full research + engineering pipeline based on  
**Plerou et al. (2001) “A Random Matrix Approach to Cross-Correlations in Financial Data”**.

## Explanation (plain language)
- Stock returns are noisy. The raw correlation matrix mixes meaningful structure (market/sector effects) with noise.
- Random Matrix Theory (RMT) tells us most eigenvalues from pure noise fall inside a predictable “bulk” range.
- We keep the informative top eigenvalue(s) and replace the noisy bulk with their average. This produces a “cleaned” correlation matrix.
- Portfolios (GMV, Risk Parity) built on cleaned correlations are often more stable and can generalize better out‑of‑sample.

## What this does
- **Data**: fetch ~200 equities (S&P 500 sub-set) over ~10 years with `yfinance`, compute daily log returns.
- **RMT analysis**: empirical correlation matrices, eigenvalue spectra, compare to the **Marchenko–Pastur** (MP) law, **IPR** (inverse participation ratio), market-mode regression residuals, eigenvector distribution checks, **overlap matrices** across windows.
- **Cleaning**: eigenvalue clipping (noise bulk → average eigenvalue), optional shrinkage.
- **Portfolios**: Global Minimum Variance (GMV), Risk Parity, with **raw vs RMT-cleaned** correlations.
- **Backtest**: rolling out-of-sample evaluation with turnover, Sharpe, drawdown, etc.

## How to run
1. **Create venv and install deps**

   - Windows (PowerShell):
     ```powershell
     py -m venv .venv
     .\\.venv\\Scripts\\Activate.ps1
     python -m pip install --upgrade pip
     pip install -r requirements.txt
     ```

   - macOS/Linux (bash/zsh):
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     python -m pip install --upgrade pip
     pip install -r requirements.txt
     ```

   - Windows (Git Bash):
     ```bash
     python -m venv .venv
     source .venv/Scripts/activate
     python -m pip install --upgrade pip
     pip install -r requirements.txt
     ```

2. See the full walkthrough in `GUIDE.md` (overview, setup, CLI + notebooks, and plots).

Quick CLI demo that saves plots to `outputs/`:
```bash
python scripts/run_project.py --window 504 --rebalance 21 --methods gmv riskparity --tc_bps 1.0 --use_cleaned both
```
