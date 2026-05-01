"""
QuantStrat - ML Trading Strategy Module (INFERENCE ONLY)

This module uses ONLY the pre-trained model exported from the notebook.
No training is performed here — all research work lives in the notebook.

Architecture:
- Notebook (AAPL_optimized_strategy.ipynb) → Research, training, optimisation
- QuantStrat (strategy.py) → Production, inference, real-time signals
"""

import yfinance as yf
import pandas as pd
import numpy as np
import talib
import os
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION (loaded from the notebook)
# ═══════════════════════════════════════════════════════════════════════════════

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
CONFIG_PATH = os.path.join(MODELS_DIR, 'optimal_config.json')
MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.joblib')
FEATURES_PATH = os.path.join(MODELS_DIR, 'feature_cols.json')

# Global variable for the model (loaded once)
_MODEL = None
_CONFIG = None
_FEATURE_COLS = None


def _load_artifacts():
    """Load the model, config and features from the notebook export."""
    global _MODEL, _CONFIG, _FEATURE_COLS
    
    if _MODEL is not None:
        return True  # Already loaded
    
    # Check that files exist
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"❌ Model not found: {MODEL_PATH}\n"
            "   → Run the notebook and the 'EXPORT TO QUANTSTRAT' cell first"
        )
    
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"❌ Config not found: {CONFIG_PATH}")
    
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"❌ Features not found: {FEATURES_PATH}")
    
    # Load
    _MODEL = joblib.load(MODEL_PATH)
    with open(CONFIG_PATH, 'r') as f:
        _CONFIG = json.load(f)
    with open(FEATURES_PATH, 'r') as f:
        _FEATURE_COLS = json.load(f)
    
    print(f"✅ Model loaded: {MODEL_PATH}")
    print(f"   Mode: {_CONFIG['MODE']}, Long: {_CONFIG['LONG_THRESHOLD']}, Short: {_CONFIG['SHORT_THRESHOLD']}")
    
    return True


def get_config():
    """Return the configuration loaded from the notebook."""
    _load_artifacts()
    return _CONFIG.copy()


def get_model():
    """Return the pre-trained model."""
    _load_artifacts()
    return _MODEL


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (identical to the notebook)
# ═══════════════════════════════════════════════════════════════════════════════

def create_features(df):
    """Compute all technical features for the model."""
    df = df.copy()
    close = df['Close'].values
    
    # Returns
    df['returns'] = df['Close'].pct_change(fill_method=None)
    for d in [5, 10, 22, 63]:
        df[f'returns_{d}d'] = df['Close'].pct_change(d, fill_method=None)
    
    # Volatility
    for w in [10, 22, 63]:
        df[f'volatility_{w}d'] = df['returns'].rolling(window=w).std()
    df['vol_ratio'] = df['volatility_10d'] / df['volatility_63d']
    
    # Moving Averages
    for p in [10, 22, 50, 200]:
        df[f'SMA_{p}'] = talib.SMA(close, timeperiod=p)
        df[f'price_to_SMA{p}'] = df['Close'] / df[f'SMA_{p}'] - 1
    
    # Momentum
    df['RSI_14'] = talib.RSI(close, timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(close)
    df['ROC_10'] = talib.ROC(close, timeperiod=10)
    
    # Z-Score
    for w in [22, 63]:
        roll_mean = df['Close'].rolling(w).mean()
        roll_std = df['Close'].rolling(w).std()
        df[f'zscore_{w}'] = (df['Close'] - roll_mean) / roll_std
    
    # Bollinger
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(close, timeperiod=20)
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Market regime (price above/below 200-day SMA)
    df['regime_sma'] = (df['Close'] > df['SMA_200']).astype(int)
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _predict_with_thresholds(model, X, long_th, short_th):
    """Generate predictions using probability thresholds."""
    proba = model.predict_proba(X)
    classes = model.classes_
    long_idx = np.where(classes == 1)[0][0]
    short_idx = np.where(classes == -1)[0][0]
    
    preds = np.zeros(len(proba))
    for i in range(len(proba)):
        if proba[i, long_idx] >= long_th:
            preds[i] = 1
        elif proba[i, short_idx] >= short_th:
            preds[i] = -1
    return preds.astype(int), proba


def _apply_regime_filter(predictions, regime, mode='standard'):
    """Apply a regime filter with different operating modes."""
    filtered = np.array(predictions).copy()
    regime = np.array(regime)
    
    for i in range(len(filtered)):
        is_bull = regime[i] == 1
        if is_bull:
            if mode == 'standard':
                if filtered[i] == -1:
                    filtered[i] = 0
            elif mode == 'aggressive':
                if filtered[i] == -1:
                    filtered[i] = 0
                elif filtered[i] == 0:
                    filtered[i] = 1
            elif mode == 'ultra':
                if filtered[i] != 1:
                    filtered[i] = 1 if filtered[i] == 0 else 0
                if filtered[i] == 0:
                    filtered[i] = 1
            elif mode == 'full_long':
                filtered[i] = 1
    return filtered.astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN API: REAL-TIME SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════

def get_current_signal(ticker: str = "AAPL", lookback: str = "1y") -> dict:
    """
    Get the current trading signal for a ticker.

    USES THE PRE-TRAINED MODEL FROM THE NOTEBOOK.

    Args:
        ticker: Stock symbol
        lookback: Data period to load (for computing features)

    Returns:
        dict with:
        - signal: 1 (BUY), 0 (HOLD/CASH), -1 (SELL)
        - signal_text: 'BUY', 'HOLD', 'SELL'
        - confidence: Signal probability
        - price: Current price
        - regime: 'BULL' or 'BEAR'
        - date: Signal date
    """
    _load_artifacts()
    
    # 1. Load recent data
    df = yf.Ticker(ticker).history(period=lookback)
    if len(df) < 250:
        return {'error': f'Not enough data for {ticker}'}
    
    df.ffill(inplace=True)
    
    # 2. Compute features
    df = create_features(df)
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    
    # 3. Prepare features for the model
    feature_cols = [c for c in _FEATURE_COLS if c in df.columns]
    X = df[feature_cols].iloc[[-1]]  # Last row only
    
    # 4. Predict
    preds_raw, proba = _predict_with_thresholds(
        _MODEL, X, 
        _CONFIG['LONG_THRESHOLD'],
        _CONFIG['SHORT_THRESHOLD']
    )
    
    # 5. Apply regime filter
    regime = df['regime_sma'].iloc[-1]
    signal = _apply_regime_filter(preds_raw, [regime], _CONFIG['MODE'])[0]
    
    # 6. Extract confidence
    classes = _MODEL.classes_
    if signal == 1:
        confidence = proba[0, np.where(classes == 1)[0][0]]
    elif signal == -1:
        confidence = proba[0, np.where(classes == -1)[0][0]]
    else:
        confidence = proba[0, np.where(classes == 0)[0][0]]
    
    # 7. Format result
    signal_text = {1: 'BUY', 0: 'HOLD', -1: 'SELL'}[signal]
    
    return {
        'ticker': ticker,
        'signal': int(signal),
        'signal_text': signal_text,
        'confidence': round(float(confidence), 4),
        'price': round(float(df['Close'].iloc[-1]), 2),
        'regime': 'BULL' if regime == 1 else 'BEAR',
        'date': df.index[-1].strftime('%Y-%m-%d'),
        'config': {
            'mode': _CONFIG['MODE'],
            'long_threshold': _CONFIG['LONG_THRESHOLD'],
            'short_threshold': _CONFIG['SHORT_THRESHOLD']
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# API: BACKTEST (with pre-trained model)
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(ticker: str = "AAPL", period: str = "2y", 
                 transaction_cost: float = 0.001,
                 use_notebook_period: bool = True) -> dict:
    """
    Run a backtest using the pre-trained model from the notebook.

    NO TRAINING — uses the model exported from the notebook.

    Args:
        ticker: Stock symbol
        period: Data period to load (if use_notebook_period=False)
        transaction_cost: Cost per transaction
        use_notebook_period: If True, uses TEST_START_DATE/TEST_END_DATE from notebook config

    Returns:
        dict with portfolio metrics and history
    """
    _load_artifacts()
    
    # 1. Load data
    # Load extra history to have enough for feature computation
    df = yf.Ticker(ticker).history(period="6y")
    if len(df) < 100:
        return {'error': 'Not enough data'}
    
    df.ffill(inplace=True)
    df = create_features(df)
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    
    # 2. Filter on the notebook's test period (if available and requested)
    test_start = _CONFIG.get('TEST_START_DATE')
    test_end = _CONFIG.get('TEST_END_DATE')
    
    if use_notebook_period and test_start and test_end:
        # Filter to only use the notebook's test period
        df = df.loc[test_start:test_end]
        print(f"📊 Backtest on notebook period: {test_start} → {test_end}")
    else:
        # Use the classic period parameter
        df = df.last(period)
        print(f"📊 Backtest on period: {period}")
    
    # 2. Prepare features
    feature_cols = [c for c in _FEATURE_COLS if c in df.columns]
    X = df[feature_cols]
    prices = df['Close'].values
    regime = df['regime_sma'].values
    dates = df.index.strftime('%Y-%m-%d').tolist()
    
    # 3. Predict
    preds_raw, _ = _predict_with_thresholds(
        _MODEL, X, 
        _CONFIG['LONG_THRESHOLD'],
        _CONFIG['SHORT_THRESHOLD']
    )
    signals = _apply_regime_filter(preds_raw, regime, _CONFIG['MODE'])
    
    # 4. Simulate portfolio
    initial_capital = 10000.0
    cash = initial_capital
    shares = 0.0
    portfolio_values = []
    trades = []
    prev_signal = 0
    
    for i in range(len(signals)):
        price = prices[i]
        signal = signals[i]
        
        if signal != prev_signal:
            if signal == 1 and shares == 0:
                shares = (cash * (1 - transaction_cost)) / price
                cash = 0.0
                trades.append({'date': dates[i], 'type': 'BUY', 'price': round(price, 2)})
            elif signal in [0, -1] and shares > 0:
                cash = shares * price * (1 - transaction_cost)
                shares = 0.0
                trades.append({'date': dates[i], 'type': 'SELL', 'price': round(price, 2)})
        
        portfolio_values.append(cash + shares * price)
        prev_signal = signal
    
    # 5. Buy & Hold
    bh_shares = (initial_capital * (1 - transaction_cost)) / prices[0]
    bh_values = [bh_shares * p for p in prices]
    
    # 6. Metrics
    portfolio_norm = np.array(portfolio_values) / initial_capital
    bh_norm = np.array(bh_values) / initial_capital
    
    strategy_returns = np.diff(portfolio_norm, prepend=portfolio_norm[0]) / np.maximum(
        np.concatenate([[1.0], portfolio_norm[:-1]]), 1e-10
    )
    bh_returns = np.diff(bh_norm, prepend=bh_norm[0]) / np.maximum(
        np.concatenate([[1.0], bh_norm[:-1]]), 1e-10
    )
    
    def sharpe(returns, ann=50):
        if np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(ann)
    
    def max_dd(values):
        peak = np.maximum.accumulate(values)
        dd = (values - peak) / peak
        return np.min(dd)
    
    # Monthly PnL
    monthly_df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'returns': strategy_returns
    })
    monthly_df['month'] = monthly_df['date'].dt.to_period('M').astype(str)
    monthly_pnl = monthly_df.groupby('month')['returns'].sum().to_dict()
    monthly_pnl = {k: round(v * 100, 2) for k, v in monthly_pnl.items()}
    
    return {
        'dates': dates,
        'portfolio_values': [round(v / initial_capital, 4) for v in portfolio_values],
        'buyhold_values': [round(v / initial_capital, 4) for v in bh_values],
        'trades': trades,
        'num_trades': len(trades),
        'total_pnl': round((portfolio_values[-1] / initial_capital - 1) * 100, 2),
        'total_pnl_buyhold': round((bh_values[-1] / initial_capital - 1) * 100, 2),
        'sharpe_ratio': round(sharpe(strategy_returns), 4),
        'sharpe_buyhold': round(sharpe(bh_returns), 4),
        'max_drawdown': round(max_dd(portfolio_norm) * 100, 2),
        'max_drawdown_buyhold': round(max_dd(bh_norm) * 100, 2),
        'monthly_pnl': monthly_pnl,
        'period': f"{dates[0]} → {dates[-1]}",
        'training_info': {
            'train_period': 'Notebook (pre-trained)',
            'test_period': f"{dates[0]} → {dates[-1]}",
            'train_samples': 'N/A',
            'test_samples': len(dates),
            'model_source': 'pretrained'
        },
        'config': {
            'mode': _CONFIG['MODE'],
            'long_threshold': _CONFIG['LONG_THRESHOLD'],
            'short_threshold': _CONFIG['SHORT_THRESHOLD'],
            'transaction_cost': transaction_cost
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
"""
WHY MONTE CARLO IS RELEVANT FOR OUR ML STRATEGY?

1. UNCERTAINTY QUANTIFICATION
   - Financial markets are inherently stochastic
   - Monte Carlo generates thousands of plausible future scenarios
   - Shows how our ML strategy reacts to different market regimes

2. ROBUSTNESS EVALUATION
   - Our model was trained on historical data
   - Monte Carlo tests whether the strategy performs under varied conditions
   - Identifies scenarios where the strategy fails (crashes, rallies)

3. RETURN DISTRIBUTION
   - Instead of a single backtest, we obtain a full distribution
   - Computes probabilities: P(profit), P(beat B&H), VaR, etc.
   - Statistically reliable measure of expected gain

4. GBM MODEL (Geometric Brownian Motion)
   - dS = μS·dt + σS·dW where μ=drift, σ=volatility, dW=Brownian motion
   - Captures the log-normal nature of equity returns
   - Parameters estimated on real historical data (μ, σ)

5. APPLICATION TO OUR STRATEGY
   - For each simulated trajectory, we recompute technical features
   - The ML model generates signals (BUY/HOLD/SELL) on simulated prices
   - We compare the ML strategy vs Buy&Hold on each trajectory
"""

def run_monte_carlo(ticker: str = "AAPL", 
                    horizon: str = "6mo",
                    n_simulations: int = 500,
                    transaction_cost: float = 0.001,
                    seed: int = None) -> dict:
    """
    Run a Monte Carlo simulation to evaluate the ML strategy.

    Uses the GBM (Geometric Brownian Motion) model to simulate future price
    trajectories, then applies the ML strategy on each trajectory to obtain
    a distribution of returns.

    Args:
        ticker: Stock symbol
        horizon: Simulation horizon ('1mo', '2mo', '6mo', '1y', '5y')
        n_simulations: Number of trajectories to simulate
        transaction_cost: Cost per transaction
        seed: Random seed (for reproducibility)

    Returns:
        dict with:
        - trajectories: Sample of trajectories (for visualisation)
        - strategy_returns: Strategy return distribution
        - buyhold_returns: Buy-and-hold return distribution
        - statistics: Aggregated metrics (mean, median, VaR, etc.)
        - probabilities: P(profit), P(beat B&H), etc.
    """
    _load_artifacts()
    
    if seed is not None:
        np.random.seed(seed)
    
    # Mapping horizon → nombre de jours de trading
    horizon_days = {
        '1mo': 21,
        '2mo': 42,
        '6mo': 126,
        '1y': 252,
        '5y': 1260
    }
    
    if horizon not in horizon_days:
        return {'error': f'Invalid horizon. Choices: {list(horizon_days.keys())}'}
    
    n_days = horizon_days[horizon]
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1. LOAD HISTORICAL DATA TO ESTIMATE GBM PARAMETERS
    # ─────────────────────────────────────────────────────────────────────────
    
    # Load 2 years of history to estimate μ and σ
    df_hist = yf.Ticker(ticker).history(period="2y")
    if len(df_hist) < 252:
        return {'error': 'Not enough historical data'}
    
    df_hist.ffill(inplace=True)
    
    # Compute daily log returns
    log_returns = np.log(df_hist['Close'] / df_hist['Close'].shift(1)).dropna()
    
    # GBM parameters estimated on historical data
    mu_daily = log_returns.mean()      # Daily drift
    sigma_daily = log_returns.std()    # Daily volatility
    
    # Initial price = last known price
    S0 = df_hist['Close'].iloc[-1]
    last_date = df_hist.index[-1]
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2. GENERATE MONTE CARLO TRAJECTORIES (GBM)
    # ─────────────────────────────────────────────────────────────────────────
    
    dt = 1  # 1 jour
    
    # Random shocks matrix: (n_simulations, n_days)
    Z = np.random.standard_normal((n_simulations, n_days))
    
    # GBM: S(t+1) = S(t) * exp((μ - σ²/2)*dt + σ*√dt*Z)
    drift = (mu_daily - 0.5 * sigma_daily**2) * dt
    diffusion = sigma_daily * np.sqrt(dt) * Z
    
    # Cumulative log returns
    log_returns_sim = drift + diffusion
    log_price_paths = np.cumsum(log_returns_sim, axis=1)
    
    # Trajectoires de prix
    price_paths = S0 * np.exp(log_price_paths)
    
    # Ajouter le prix initial
    price_paths = np.column_stack([np.full(n_simulations, S0), price_paths])
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3. APPLIQUER LA STRATÉGIE ML SUR CHAQUE TRAJECTOIRE
    # ─────────────────────────────────────────────────────────────────────────
    
    strategy_returns = []
    buyhold_returns = []
    
    # Pour chaque simulation, on calcule la performance
    for i in range(n_simulations):
        prices = price_paths[i]
        
        # Daily returns for this trajectory
        daily_rets = np.diff(prices) / prices[:-1]
        
        # Generate simplified signals based on regime
        # (Use an approximate SMA on the simulated trajectory)
        sma_200 = pd.Series(prices).rolling(min(200, len(prices)//2)).mean().values
        regime = (prices > sma_200).astype(int)
        
        # Simplified signal based on regime and trend
        # In aggressive mode: LONG in bull market, otherwise CASH
        mode = _CONFIG['MODE']
        
        signals = np.zeros(len(prices))
        for j in range(1, len(prices)):
            if regime[j] == 1:  # Bull market
                if mode in ['aggressive', 'ultra', 'full_long']:
                    signals[j] = 1  # LONG
                else:
                    # Standard: look at short-term trend
                    if j > 10:
                        short_trend = prices[j] / prices[j-10] - 1
                        signals[j] = 1 if short_trend > 0 else 0
                    else:
                        signals[j] = 1
            else:  # Bear market
                signals[j] = 0  # CASH
        
        # Simulate strategy portfolio
        cash = 10000.0
        shares = 0.0
        prev_signal = 0
        
        for j in range(1, len(prices)):
            signal = signals[j]
            price = prices[j]
            
            if signal != prev_signal:
                if signal == 1 and shares == 0:
                    shares = (cash * (1 - transaction_cost)) / price
                    cash = 0
                elif signal == 0 and shares > 0:
                    cash = shares * price * (1 - transaction_cost)
                    shares = 0
            
            prev_signal = signal
        
        # Final strategy value
        final_value_strat = cash + shares * prices[-1]
        ret_strat = (final_value_strat / 10000.0 - 1) * 100
        strategy_returns.append(ret_strat)
        
        # Valeur finale B&H
        ret_bh = (prices[-1] / prices[0] - 1) * 100
        buyhold_returns.append(ret_bh)
    
    strategy_returns = np.array(strategy_returns)
    buyhold_returns = np.array(buyhold_returns)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4. COMPUTE STATISTICS
    # ─────────────────────────────────────────────────────────────────────────
    
    def percentile_stats(arr):
        return {
            'mean': round(float(np.mean(arr)), 2),
            'median': round(float(np.median(arr)), 2),
            'std': round(float(np.std(arr)), 2),
            'min': round(float(np.min(arr)), 2),
            'max': round(float(np.max(arr)), 2),
            'p5': round(float(np.percentile(arr, 5)), 2),   # 95% VaR
            'p25': round(float(np.percentile(arr, 25)), 2),
            'p75': round(float(np.percentile(arr, 75)), 2),
            'p95': round(float(np.percentile(arr, 95)), 2),
        }
    
    # Probabilities
    prob_profit_strat = float(np.mean(strategy_returns > 0))
    prob_profit_bh = float(np.mean(buyhold_returns > 0))
    prob_beat_bh = float(np.mean(strategy_returns > buyhold_returns))
    
    # Mean outperformance
    outperformance = strategy_returns - buyhold_returns
    
    # Select a sample of trajectories for visualisation
    n_display = min(50, n_simulations)
    sample_indices = np.random.choice(n_simulations, n_display, replace=False)
    
    # Generate future dates
    future_dates = pd.date_range(start=last_date, periods=n_days+1, freq='B')
    dates_str = [d.strftime('%Y-%m-%d') for d in future_dates]
    
    return {
        'horizon': horizon,
        'n_simulations': n_simulations,
        'n_days': n_days,
        'current_price': round(float(S0), 2),
        'last_date': last_date.strftime('%Y-%m-%d'),
        
        # GBM parameters estimated on historical data
        'gbm_params': {
            'mu_annual': round(float(mu_daily * 252 * 100), 2),  # % annualised
            'sigma_annual': round(float(sigma_daily * np.sqrt(252) * 100), 2),  # % annualised
            'mu_daily': round(float(mu_daily * 100), 4),
            'sigma_daily': round(float(sigma_daily * 100), 4),
        },
        
        # Trajectories (sample for visualisation)
        'dates': dates_str,
        'trajectories': [
            [round(float(p), 2) for p in price_paths[i]] 
            for i in sample_indices
        ],
        
        # Return distribution
        'strategy_returns': [round(float(r), 2) for r in strategy_returns],
        'buyhold_returns': [round(float(r), 2) for r in buyhold_returns],
        
        # Statistics
        'statistics': {
            'strategy': percentile_stats(strategy_returns),
            'buyhold': percentile_stats(buyhold_returns),
            'outperformance': percentile_stats(outperformance),
        },
        
        # Probabilities
        'probabilities': {
            'profit_strategy': round(prob_profit_strat * 100, 1),
            'profit_buyhold': round(prob_profit_bh * 100, 1),
            'beat_buyhold': round(prob_beat_bh * 100, 1),
        },
        
        # Config used
        'config': {
            'mode': _CONFIG['MODE'],
            'transaction_cost': transaction_cost,
        },
        
        # Metadata
        'method': 'simplified_regime'  # Indicates the simplified version
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO SIMULATION WITH FULL ML MODEL
# ═══════════════════════════════════════════════════════════════════════════════
"""
IMPROVED VERSION: Uses the real RandomForest model on each trajectory.

Unlike the simplified version which only uses the SMA regime,
this version:
1. Computes ALL technical features (RSI, MACD, Bollinger, etc.)
2. Uses the pre-trained ML model to predict
3. Applies probability thresholds and regime filters

Slower (~10x) but much more realistic.
"""

def run_monte_carlo_ml(ticker: str = "AAPL", 
                       horizon: str = "6mo",
                       n_simulations: int = 100,  # Fewer by default as ML is slower
                       transaction_cost: float = 0.001,
                       seed: int = None) -> dict:
    """
    Monte Carlo with the real ML model (RandomForest).

    For each simulated trajectory:
    1. Concatenates recent history + simulated prices
    2. Computes all technical features
    3. Uses the ML model to generate signals
    4. Simulates the portfolio

    Slower but more realistic than run_monte_carlo().
    """
    _load_artifacts()
    
    if seed is not None:
        np.random.seed(seed)
    
    # Mapping horizon
    horizon_days = {
        '1mo': 21,
        '2mo': 42,
        '6mo': 126,
        '1y': 252,
        '5y': 1260
    }
    
    if horizon not in horizon_days:
        return {'error': f'Invalid horizon. Choices: {list(horizon_days.keys())}'}
    
    n_days = horizon_days[horizon]
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1. LOAD HISTORY (enough for SMA_200 + buffer)
    # ─────────────────────────────────────────────────────────────────────────
    
    df_hist = yf.Ticker(ticker).history(period="2y")
    if len(df_hist) < 300:
        return {'error': 'Not enough historical data (need 300+ days)'}
    
    df_hist.ffill(inplace=True)
    
    # Keep the last 250 days for history
    hist_days = 250
    df_hist = df_hist.iloc[-hist_days:]
    
    # GBM parameters
    log_returns = np.log(df_hist['Close'] / df_hist['Close'].shift(1)).dropna()
    mu_daily = log_returns.mean()
    sigma_daily = log_returns.std()
    
    S0 = df_hist['Close'].iloc[-1]
    last_date = df_hist.index[-1]
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2. GENERATE GBM TRAJECTORIES
    # ─────────────────────────────────────────────────────────────────────────
    
    Z = np.random.standard_normal((n_simulations, n_days))
    drift = (mu_daily - 0.5 * sigma_daily**2)
    diffusion = sigma_daily * Z
    log_returns_sim = drift + diffusion
    log_price_paths = np.cumsum(log_returns_sim, axis=1)
    price_paths = S0 * np.exp(log_price_paths)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3. APPLY THE REAL ML MODEL ON EACH TRAJECTORY
    # ─────────────────────────────────────────────────────────────────────────
    
    strategy_returns = []
    buyhold_returns = []
    n_trades_list = []
    
    # Future dates
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                  periods=n_days, freq='B')
    
    for i in range(n_simulations):
        # Build a DataFrame with history + simulation
        sim_prices = price_paths[i]
        
        # Build combined DataFrame
        df_sim = pd.DataFrame({
            'Close': np.concatenate([df_hist['Close'].values, sim_prices]),
            'Open': np.concatenate([df_hist['Open'].values, sim_prices * 0.999]),  # Approximation
            'High': np.concatenate([df_hist['High'].values, sim_prices * 1.005]),
            'Low': np.concatenate([df_hist['Low'].values, sim_prices * 0.995]),
            'Volume': np.concatenate([df_hist['Volume'].values, 
                                      np.full(n_days, df_hist['Volume'].mean())])
        })
        
        # Compute features
        df_feat = create_features(df_sim)
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        # Keep only the simulated part
        df_future = df_feat.iloc[hist_days:]
        
        # Prepare for the model
        feature_cols = [c for c in _FEATURE_COLS if c in df_future.columns]
        X = df_future[feature_cols].dropna()
        
        if len(X) == 0:
            # Not enough data, fall back to B&H
            ret_bh = (sim_prices[-1] / sim_prices[0] - 1) * 100
            strategy_returns.append(ret_bh)
            buyhold_returns.append(ret_bh)
            n_trades_list.append(0)
            continue
        
        # Predict with the ML model
        preds_raw, proba = _predict_with_thresholds(
            _MODEL, X,
            _CONFIG['LONG_THRESHOLD'],
            _CONFIG['SHORT_THRESHOLD']
        )
        
        # Apply regime filter
        regime = df_future.loc[X.index, 'regime_sma'].values
        signals = _apply_regime_filter(preds_raw, regime, _CONFIG['MODE'])
        
        # Prices corresponding to signals
        prices_for_sim = df_future.loc[X.index, 'Close'].values
        
        # Simulate portfolio
        cash = 10000.0
        shares = 0.0
        prev_signal = 0
        n_trades = 0
        
        for j in range(len(signals)):
            signal = signals[j]
            price = prices_for_sim[j]
            
            if signal != prev_signal:
                if signal == 1 and shares == 0:
                    shares = (cash * (1 - transaction_cost)) / price
                    cash = 0
                    n_trades += 1
                elif signal in [0, -1] and shares > 0:
                    cash = shares * price * (1 - transaction_cost)
                    shares = 0
                    n_trades += 1
            
            prev_signal = signal
        
        final_value = cash + shares * prices_for_sim[-1]
        ret_strat = (final_value / 10000.0 - 1) * 100
        ret_bh = (sim_prices[-1] / sim_prices[0] - 1) * 100
        
        strategy_returns.append(ret_strat)
        buyhold_returns.append(ret_bh)
        n_trades_list.append(n_trades)
    
    strategy_returns = np.array(strategy_returns)
    buyhold_returns = np.array(buyhold_returns)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4. STATISTIQUES
    # ─────────────────────────────────────────────────────────────────────────
    
    def percentile_stats(arr):
        return {
            'mean': round(float(np.mean(arr)), 2),
            'median': round(float(np.median(arr)), 2),
            'std': round(float(np.std(arr)), 2),
            'min': round(float(np.min(arr)), 2),
            'max': round(float(np.max(arr)), 2),
            'p5': round(float(np.percentile(arr, 5)), 2),
            'p25': round(float(np.percentile(arr, 25)), 2),
            'p75': round(float(np.percentile(arr, 75)), 2),
            'p95': round(float(np.percentile(arr, 95)), 2),
        }
    
    prob_profit_strat = float(np.mean(strategy_returns > 0))
    prob_profit_bh = float(np.mean(buyhold_returns > 0))
    prob_beat_bh = float(np.mean(strategy_returns > buyhold_returns))
    outperformance = strategy_returns - buyhold_returns
    
    # Sample of trajectories
    n_display = min(30, n_simulations)
    sample_indices = np.random.choice(n_simulations, n_display, replace=False)
    
    dates_str = [last_date.strftime('%Y-%m-%d')] + [d.strftime('%Y-%m-%d') for d in future_dates]
    
    # Reconstruct full trajectories for display
    trajectories_display = []
    for idx in sample_indices:
        traj = np.concatenate([[S0], price_paths[idx]])
        trajectories_display.append([round(float(p), 2) for p in traj])
    
    return {
        'horizon': horizon,
        'n_simulations': n_simulations,
        'n_days': n_days,
        'current_price': round(float(S0), 2),
        'last_date': last_date.strftime('%Y-%m-%d'),
        
        'gbm_params': {
            'mu_annual': round(float(mu_daily * 252 * 100), 2),
            'sigma_annual': round(float(sigma_daily * np.sqrt(252) * 100), 2),
            'mu_daily': round(float(mu_daily * 100), 4),
            'sigma_daily': round(float(sigma_daily * 100), 4),
        },
        
        'dates': dates_str,
        'trajectories': trajectories_display,
        
        'strategy_returns': [round(float(r), 2) for r in strategy_returns],
        'buyhold_returns': [round(float(r), 2) for r in buyhold_returns],
        
        'statistics': {
            'strategy': percentile_stats(strategy_returns),
            'buyhold': percentile_stats(buyhold_returns),
            'outperformance': percentile_stats(outperformance),
        },
        
        'probabilities': {
            'profit_strategy': round(prob_profit_strat * 100, 1),
            'profit_buyhold': round(prob_profit_bh * 100, 1),
            'beat_buyhold': round(prob_beat_bh * 100, 1),
        },
        
        'trading_stats': {
            'avg_trades': round(float(np.mean(n_trades_list)), 1),
            'max_trades': int(np.max(n_trades_list)),
            'min_trades': int(np.min(n_trades_list)),
        },
        
        'config': {
            'mode': _CONFIG['MODE'],
            'long_threshold': _CONFIG['LONG_THRESHOLD'],
            'short_threshold': _CONFIG['SHORT_THRESHOLD'],
            'transaction_cost': transaction_cost,
        },
        
        'method': 'full_ml_model'  # Full ML model version
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO SIMULATION WITH GARCH (Time-Varying Volatility)
# ═══════════════════════════════════════════════════════════════════════════════
"""
GARCH (Generalized Autoregressive Conditional Heteroskedasticity)

WHY GARCH IS MORE REALISTIC THAN GBM:

1. TIME-VARYING VOLATILITY
   - GBM: σ is constant over time
   - GARCH: σ_t varies based on past observations

2. VOLATILITY CLUSTERING
   - High-volatility periods tend to be followed by high-volatility periods
   - Calm periods tend to be followed by calm periods
   - This is a well-known stylised fact of financial markets

3. GARCH(1,1) MODEL
   σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

   where:
   - ω (omega) = base variance level
   - α (alpha) = reaction to shocks (news)
   - β (beta) = volatility persistence
   - α + β < 1 for stationarity (often ~0.98)

4. IMPACT ON THE ML STRATEGY
   - The ML model uses volatility as a feature
   - GARCH generates realistic volatility patterns
   - Allows testing whether the strategy adapts to different regimes
"""

def run_monte_carlo_garch(ticker: str = "AAPL", 
                          horizon: str = "6mo",
                          n_simulations: int = 100,
                          transaction_cost: float = 0.001,
                          seed: int = None) -> dict:
    """
    Monte Carlo with GARCH volatility model + real ML model.

    GARCH captures volatility clustering:
    - High-vol periods followed by high vol
    - More realistic than constant-volatility GBM

    Combines:
    1. GARCH simulation for volatility
    2. Real ML model (RandomForest) for signals
    """
    _load_artifacts()
    
    # Import GARCH
    try:
        from arch import arch_model
    except ImportError:
        return {'error': 'Package arch not installed. Run: pip install arch'}
    
    if seed is not None:
        np.random.seed(seed)
    
    # Mapping horizon
    horizon_days = {
        '1mo': 21,
        '2mo': 42,
        '6mo': 126,
        '1y': 252,
        '5y': 1260
    }
    
    if horizon not in horizon_days:
        return {'error': f'Invalid horizon. Choices: {list(horizon_days.keys())}'}
    
    n_days = horizon_days[horizon]
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1. LOAD HISTORY AND ESTIMATE GARCH(1,1)
    # ─────────────────────────────────────────────────────────────────────────
    
    df_hist = yf.Ticker(ticker).history(period="2y")
    if len(df_hist) < 300:
        return {'error': 'Not enough historical data (need 300+ days)'}
    
    df_hist.ffill(inplace=True)
    
    # Rendements en pourcentage (pour GARCH)
    returns_pct = df_hist['Close'].pct_change().dropna() * 100
    
    # Estimer GARCH(1,1)
    try:
        model = arch_model(returns_pct, vol='Garch', p=1, q=1, mean='Constant', rescale=False)
        result = model.fit(disp='off', show_warning=False)
        
        # Extract parameters
        mu = result.params['mu']  # Drift moyen
        omega = result.params['omega']
        alpha = result.params['alpha[1]']
        beta = result.params['beta[1]']
        
        # Last conditional variance
        last_variance = result.conditional_volatility.iloc[-1] ** 2
        
    except Exception as e:
        return {'error': f'Erreur estimation GARCH: {str(e)}'}
    
    # Check stationarity
    persistence = alpha + beta
    
    # Initial price
    S0 = df_hist['Close'].iloc[-1]
    last_date = df_hist.index[-1]
    
    # Keep history for feature computation
    hist_days = 250
    df_hist_subset = df_hist.iloc[-hist_days:]
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2. SIMULATE TRAJECTORIES WITH GARCH
    # ─────────────────────────────────────────────────────────────────────────
    
    price_paths = np.zeros((n_simulations, n_days))
    volatility_paths = np.zeros((n_simulations, n_days))
    
    for sim in range(n_simulations):
        prices = [S0]
        variances = [last_variance]
        
        for t in range(n_days):
            # Standard innovation
            z = np.random.standard_normal()
            
            # Conditional variance GARCH(1,1)
            # σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
            if t == 0:
                var_t = last_variance
                eps_prev = 0
            else:
                eps_prev = (prices[-1] / prices[-2] - 1) * 100 - mu
                var_t = omega + alpha * (eps_prev ** 2) + beta * variances[-1]
            
            var_t = max(var_t, omega)  # Floor at omega
            variances.append(var_t)
            
            # Return with conditional volatility
            sigma_t = np.sqrt(var_t)
            r_t = (mu + sigma_t * z) / 100  # Convert to decimal
            
            # New price
            new_price = prices[-1] * (1 + r_t)
            prices.append(new_price)
        
        price_paths[sim] = prices[1:]  # Exclude initial price
        volatility_paths[sim] = [np.sqrt(v) for v in variances[1:]]  # Volatility in %
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3. APPLY THE REAL ML MODEL ON EACH TRAJECTORY
    # ─────────────────────────────────────────────────────────────────────────
    
    strategy_returns = []
    buyhold_returns = []
    n_trades_list = []
    
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                  periods=n_days, freq='B')
    
    for i in range(n_simulations):
        sim_prices = price_paths[i]
        
        # Build combined DataFrame
        df_sim = pd.DataFrame({
            'Close': np.concatenate([df_hist_subset['Close'].values, sim_prices]),
            'Open': np.concatenate([df_hist_subset['Open'].values, sim_prices * 0.999]),
            'High': np.concatenate([df_hist_subset['High'].values, sim_prices * 1.005]),
            'Low': np.concatenate([df_hist_subset['Low'].values, sim_prices * 0.995]),
            'Volume': np.concatenate([df_hist_subset['Volume'].values, 
                                      np.full(n_days, df_hist_subset['Volume'].mean())])
        })
        
        # Compute features
        df_feat = create_features(df_sim)
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        # Keep only the simulated part
        df_future = df_feat.iloc[hist_days:]
        
        # Prepare for the model
        feature_cols = [c for c in _FEATURE_COLS if c in df_future.columns]
        X = df_future[feature_cols].dropna()
        
        if len(X) == 0:
            ret_bh = (sim_prices[-1] / sim_prices[0] - 1) * 100
            strategy_returns.append(ret_bh)
            buyhold_returns.append(ret_bh)
            n_trades_list.append(0)
            continue
        
        # Predict with the ML model
        preds_raw, proba = _predict_with_thresholds(
            _MODEL, X,
            _CONFIG['LONG_THRESHOLD'],
            _CONFIG['SHORT_THRESHOLD']
        )
        
        # Apply regime filter
        regime = df_future.loc[X.index, 'regime_sma'].values
        signals = _apply_regime_filter(preds_raw, regime, _CONFIG['MODE'])
        
        # Prices corresponding to signals
        prices_for_sim = df_future.loc[X.index, 'Close'].values
        
        # Simulate portfolio
        cash = 10000.0
        shares = 0.0
        prev_signal = 0
        n_trades = 0
        
        for j in range(len(signals)):
            signal = signals[j]
            price = prices_for_sim[j]
            
            if signal != prev_signal:
                if signal == 1 and shares == 0:
                    shares = (cash * (1 - transaction_cost)) / price
                    cash = 0
                    n_trades += 1
                elif signal in [0, -1] and shares > 0:
                    cash = shares * price * (1 - transaction_cost)
                    shares = 0
                    n_trades += 1
            
            prev_signal = signal
        
        final_value = cash + shares * prices_for_sim[-1]
        ret_strat = (final_value / 10000.0 - 1) * 100
        ret_bh = (sim_prices[-1] / sim_prices[0] - 1) * 100
        
        strategy_returns.append(ret_strat)
        buyhold_returns.append(ret_bh)
        n_trades_list.append(n_trades)
    
    strategy_returns = np.array(strategy_returns)
    buyhold_returns = np.array(buyhold_returns)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4. STATISTIQUES
    # ─────────────────────────────────────────────────────────────────────────
    
    def percentile_stats(arr):
        return {
            'mean': round(float(np.mean(arr)), 2),
            'median': round(float(np.median(arr)), 2),
            'std': round(float(np.std(arr)), 2),
            'min': round(float(np.min(arr)), 2),
            'max': round(float(np.max(arr)), 2),
            'p5': round(float(np.percentile(arr, 5)), 2),
            'p25': round(float(np.percentile(arr, 25)), 2),
            'p75': round(float(np.percentile(arr, 75)), 2),
            'p95': round(float(np.percentile(arr, 95)), 2),
        }
    
    prob_profit_strat = float(np.mean(strategy_returns > 0))
    prob_profit_bh = float(np.mean(buyhold_returns > 0))
    prob_beat_bh = float(np.mean(strategy_returns > buyhold_returns))
    outperformance = strategy_returns - buyhold_returns
    
    # Sample of trajectories
    n_display = min(30, n_simulations)
    sample_indices = np.random.choice(n_simulations, n_display, replace=False)
    
    dates_str = [last_date.strftime('%Y-%m-%d')] + [d.strftime('%Y-%m-%d') for d in future_dates]
    
    trajectories_display = []
    for idx in sample_indices:
        traj = np.concatenate([[S0], price_paths[idx]])
        trajectories_display.append([round(float(p), 2) for p in traj])
    
    # Average volatility across trajectories (informational)
    avg_volatility = np.mean(volatility_paths, axis=1).mean()
    
    return {
        'horizon': horizon,
        'n_simulations': n_simulations,
        'n_days': n_days,
        'current_price': round(float(S0), 2),
        'last_date': last_date.strftime('%Y-%m-%d'),
        
        # GARCH parameters
        'garch_params': {
            'mu': round(float(mu), 4),
            'omega': round(float(omega), 6),
            'alpha': round(float(alpha), 4),
            'beta': round(float(beta), 4),
            'persistence': round(float(persistence), 4),
            'avg_volatility_annual': round(float(avg_volatility * np.sqrt(252)), 2),
        },
        
        'dates': dates_str,
        'trajectories': trajectories_display,
        
        'strategy_returns': [round(float(r), 2) for r in strategy_returns],
        'buyhold_returns': [round(float(r), 2) for r in buyhold_returns],
        
        'statistics': {
            'strategy': percentile_stats(strategy_returns),
            'buyhold': percentile_stats(buyhold_returns),
            'outperformance': percentile_stats(outperformance),
        },
        
        'probabilities': {
            'profit_strategy': round(prob_profit_strat * 100, 1),
            'profit_buyhold': round(prob_profit_bh * 100, 1),
            'beat_buyhold': round(prob_beat_bh * 100, 1),
        },
        
        'trading_stats': {
            'avg_trades': round(float(np.mean(n_trades_list)), 1),
            'max_trades': int(np.max(n_trades_list)),
            'min_trades': int(np.min(n_trades_list)),
        },
        
        'config': {
            'mode': _CONFIG['MODE'],
            'long_threshold': _CONFIG['LONG_THRESHOLD'],
            'short_threshold': _CONFIG['SHORT_THRESHOLD'],
            'transaction_cost': transaction_cost,
        },
        
        'method': 'garch_ml_model'  # GARCH + ML
    }
