## Quant RMT Project — Beginner-Friendly Guide

This guide explains what the project does, how to set it up on Windows/macOS/Linux, and how to run it with clear visuals.

### What this project does (plain language)
- **Goal**: Build better correlation estimates between stocks using Random Matrix Theory (RMT) to filter noise, then use those estimates in simple portfolios and backtests.
- **Key ideas**:
  - Daily returns are noisy; their correlation matrix contains both information and noise.
  - RMT says most eigenvalues of a correlation matrix from noisy data fall inside a predictable "bulk" range (Marchenko–Pastur law). Eigenvalues inside this bulk are mostly noise.
  - We "clean" by replacing the noisy bulk with an average value, keeping the largest eigenvalue(s) that represent market/sector structure.
  - We then construct portfolios (GMV, Risk Parity) using the cleaned correlations and backtest them over time.

### Repository layout
- `data/raw/prices.csv`: example prices (index=dates, columns=tickers)
- `data/processed/returns_log.csv`: precomputed daily log returns
- `src/`: reusable library code
  - `rmt_analysis.py`: returns, correlations, MP bounds, eigen-decomp, clipping
  - `portfolios.py`: GMV and Risk Parity
  - `backtest.py`: rolling backtest utilities
  - `utils.py`: stats and helpers
- `configs/params.yaml/params.yaml`: dates, windows, RMT and portfolio options
- `notebooks/`: 01 data, 02 spectral analysis, 03 backtest
- `scripts/run_project.py`: command-line runner that saves plots to `outputs/`

### Setup
1) Create a virtual environment and install dependencies.

Windows (PowerShell):
```powershell
cd "B:\CV Projects\quant-rmt-project"
py -m venv .venv  # if 'py' not found, use: python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

macOS/Linux:
```bash
cd /path/to/quant-rmt-project
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option A — Run via notebooks (full exploration)
```bash
jupyter lab
```
Run in order:
1. `notebooks/01_data_preprocessing.ipynb`
2. `notebooks/02_spectral_analysis.ipynb`
3. `notebooks/03_backtest_and_portfolios.ipynb`

### Option B — One-command CLI run with saved plots
This generates results and plots into `outputs/`.

Windows (PowerShell):
```powershell
python scripts/run_project.py --window 504 --rebalance 21 --methods gmv riskparity --tc_bps 1.0 --use_cleaned both
```

macOS/Linux:
```bash
python scripts/run_project.py --window 504 --rebalance 21 --methods gmv riskparity --tc_bps 1.0 --use_cleaned both
```

Arguments:
- `--window`: in-sample window length (trading days)
- `--rebalance`: rebalance frequency (days)
- `--methods`: one or more of `gmv`, `riskparity`
- `--tc_bps`: transaction cost in basis points per rebalance
- `--use_cleaned`: `raw`, `cleaned`, or `both`

Outputs (in `outputs/`):
- `equity_curve_<strategy>.png`: equity curves
- `drawdowns_<strategy>.png`: drawdown charts
- `weights_heatmap_<strategy>.png`: time-varying weights (top holdings annotated)
- `summary.csv`: table with Sharpe, ann. return/vol, max drawdown, turnover

### Troubleshooting
- PowerShell says `py` not found: use `python` instead.
- Activation blocked: set execution policy (see above) or run the interpreter directly: `.\.venv\Scripts\python.exe script.py`
- If you see import errors, ensure you run from the repo root so relative paths to `data/` work.

### Next steps
- Change the universe or dates in `configs/params.yaml/params.yaml`.
- Toggle `use_market_regression` to remove market mode before cleaning.
- Try `leverage_cap` and `long_only` variations in the backtest.


