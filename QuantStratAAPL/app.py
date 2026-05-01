"""
QuantStrat - Flask Application for ML Backtesting
Uses the pre-trained model exported from the notebook.
"""

from flask import Flask, render_template, request, jsonify
from strategy import (
    run_backtest, get_current_signal, get_config, 
    run_monte_carlo, run_monte_carlo_ml, run_monte_carlo_garch
)

app = Flask(__name__)


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/run', methods=['POST'])
def run_backtest_route():
    """Run the backtest on the notebook's test period."""
    try:
        data = request.get_json() or {}
        transaction_cost = float(data.get('transaction_cost', 0.001))
        
        if transaction_cost < 0 or transaction_cost > 0.1:
            return jsonify({'error': 'Transaction cost must be between 0 and 10%'}), 400
        
        # Backtest on the notebook's test period (use_notebook_period=True by default)
        result = run_backtest(ticker="AAPL", transaction_cost=transaction_cost)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/signal', methods=['GET', 'POST'])
def get_signal():
    """Get the current trading signal (uses the pre-trained model)."""
    try:
        result = get_current_signal(ticker="AAPL")
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/montecarlo', methods=['POST'])
def run_monte_carlo_route():
    """Simplified Monte Carlo simulation (SMA regime only)."""
    try:
        data = request.get_json() or {}
        
        horizon = data.get('horizon', '6mo')
        n_simulations = int(data.get('n_simulations', 500))
        transaction_cost = float(data.get('transaction_cost', 0.001))
        
        valid_horizons = ['1mo', '2mo', '6mo', '1y', '5y']
        if horizon not in valid_horizons:
            return jsonify({'error': f'Invalid horizon. Choices: {valid_horizons}'}), 400
        
        if n_simulations < 100 or n_simulations > 2000:
            return jsonify({'error': 'Number of simulations must be between 100 and 2000'}), 400
        
        result = run_monte_carlo(
            ticker="AAPL",
            horizon=horizon,
            n_simulations=n_simulations,
            transaction_cost=transaction_cost
        )
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/montecarlo_ml', methods=['POST'])
def run_monte_carlo_ml_route():
    """Monte Carlo with the full ML model (GBM + RandomForest)."""
    try:
        data = request.get_json() or {}
        
        horizon = data.get('horizon', '6mo')
        n_simulations = int(data.get('n_simulations', 100))
        transaction_cost = float(data.get('transaction_cost', 0.001))
        
        valid_horizons = ['1mo', '2mo', '6mo', '1y', '5y']
        if horizon not in valid_horizons:
            return jsonify({'error': f'Invalid horizon. Choices: {valid_horizons}'}), 400
        
        if n_simulations < 50 or n_simulations > 500:
            return jsonify({'error': 'Number of simulations: 50-500 (ML is slower)'}), 400
        
        result = run_monte_carlo_ml(
            ticker="AAPL",
            horizon=horizon,
            n_simulations=n_simulations,
            transaction_cost=transaction_cost
        )
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/montecarlo_garch', methods=['POST'])
def run_monte_carlo_garch_route():
    """
    Monte Carlo with GARCH + full ML model.

    GARCH captures volatility clustering:
    - Periods of high volatility tend to be followed by high volatility
    - More realistic than constant-volatility GBM

    The most realistic of the three methods, but also the slowest.
    """
    try:
        data = request.get_json() or {}
        
        horizon = data.get('horizon', '6mo')
        n_simulations = int(data.get('n_simulations', 100))
        transaction_cost = float(data.get('transaction_cost', 0.001))
        
        valid_horizons = ['1mo', '2mo', '6mo', '1y', '5y']
        if horizon not in valid_horizons:
            return jsonify({'error': f'Invalid horizon. Choices: {valid_horizons}'}), 400
        
        if n_simulations < 50 or n_simulations > 300:
            return jsonify({'error': 'Number of simulations: 50-300 (GARCH is very slow)'}), 400
        
        result = run_monte_carlo_garch(
            ticker="AAPL",
            horizon=horizon,
            n_simulations=n_simulations,
            transaction_cost=transaction_cost
        )
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/config', methods=['GET'])
def get_model_config():
    """Get the model configuration."""
    try:
        config = get_config()
        return jsonify(config)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
