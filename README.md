# AAPL — Quantitative Trading Strategies

This repository contains research notebooks and a production-ready web application for algorithmic trading of **Apple (AAPL)** stock using Machine Learning.

---

## Repository Structure

```
AAPL/
└── QuantStratAAPL/           # Flask web application
    ├── app.py                # REST API (Flask)
    ├── strategy.py           # Core ML inference module
    ├── requirements.txt
    ├── models/               # Exported model artefacts
    │   ├── best_model.joblib
    │   ├── optimal_config.json
    │   └── feature_cols.json
    ├── static/               # Frontend assets
    │   ├── app.js
    │   └── style.css
    └── templates/
        └── index.html
```

## QuantStratAAPL — Flask Web Application

A lightweight web dashboard that serves the pre-trained model for:
- **Real-time signal**: current BUY / HOLD / SELL recommendation with confidence score and market regime.
- **Backtest**: replay the model on its original test period (dates loaded from `optimal_config.json`).
- **Monte Carlo simulations**: forward-looking risk analysis using three increasingly realistic methods.

### Architecture

```
Notebooks  ──export──▶  models/  ──load──▶  strategy.py  ◀──  app.py (Flask)
```

`strategy.py` is **inference-only** — no training happens at runtime. All model artefacts are produced once in the notebook and loaded at startup.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Dashboard UI |
| `POST` | `/run` | Run backtest on the notebook's test period |
| `GET/POST` | `/signal` | Get current real-time trading signal |
| `POST` | `/montecarlo` | Simplified Monte Carlo (SMA regime, fast) |
| `POST` | `/montecarlo_ml` | Monte Carlo with full RandomForest model |
| `POST` | `/montecarlo_garch` | Monte Carlo with GARCH + RandomForest (most realistic) |
| `GET` | `/config` | Return loaded model configuration |

### Monte Carlo Methods

Three simulation methods are available, in increasing order of realism and computational cost:

| Method | Price model | Signal generation | Typical simulations |
|---|---|---|---|
| Simplified | GBM | SMA regime only | 500 |
| Full ML | GBM | RandomForest predictions | 100 |
| GARCH + ML | GARCH(1,1) | RandomForest predictions | 100 |

**GBM** (Geometric Brownian Motion) uses $\mu$ and $\sigma$ estimated on 2 years of historical data:
$$S_{t+1} = S_t \exp\!\left(\left(\mu - \tfrac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t}\, Z\right)$$

**GARCH(1,1)** models time-varying conditional volatility to capture volatility clustering:
$$\sigma^2_t = \omega + \alpha \varepsilon^2_{t-1} + \beta \sigma^2_{t-1}$$

All simulations return return distributions, statistics (mean, median, VaR at 5%, percentiles), and probabilities such as $P(\text{profit})$ and $P(\text{beat Buy\&Hold})$.

### Running the App

```bash
cd QuantStratAAPL
pip install -r requirements.txt
python app.py
```

Then open [http://localhost:5000](http://localhost:5000).

> **Prerequisites**: the notebook `AAPL_optimized_strategy.ipynb` must be run first to export the model artefacts into `models/`.

### Dependencies

```
flask
yfinance
pandas
numpy
ta-lib
scikit-learn
joblib
arch          # optional, required for GARCH Monte Carlo
```
