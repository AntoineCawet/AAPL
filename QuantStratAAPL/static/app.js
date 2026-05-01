// ═══════════════════════════════════════════════════════════════════════════════
// QuantStrat - Application JavaScript
// ═══════════════════════════════════════════════════════════════════════════════

// Configuration Plotly
const plotlyLayout = {
    paper_bgcolor: '#0f0f0f',
    plot_bgcolor: '#0f0f0f',
    font: { family: 'Inter, sans-serif', size: 11, color: '#8a8a8a' },
    margin: { t: 30, r: 20, b: 40, l: 50 },
    xaxis: { gridcolor: 'rgba(255,255,255,0.04)', linecolor: 'rgba(255,255,255,0.08)' },
    yaxis: { gridcolor: 'rgba(255,255,255,0.04)', linecolor: 'rgba(255,255,255,0.08)' }
};
const plotlyConfig = { displayModeBar: true, responsive: true, displaylogo: false };

// ═══════════════════════════════════════════════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════════════════════════════════════════════

function switchTab(tabName) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    const idx = tabName === 'backtest' ? 1 : tabName === 'signal' ? 2 : 3;
    document.querySelector(`.tab:nth-child(${idx})`).classList.add('active');
    document.getElementById(`tab-${tabName}`).classList.add('active');
    if (tabName === 'signal') loadSignal();
}

// ═══════════════════════════════════════════════════════════════════════════════
// LOADING
// ═══════════════════════════════════════════════════════════════════════════════

function showLoading(msg) {
    document.getElementById('loadingText').textContent = msg;
    document.getElementById('loading').classList.add('active');
}

function hideLoading() {
    document.getElementById('loading').classList.remove('active');
}

// ═══════════════════════════════════════════════════════════════════════════════
// BACKTEST
// ═══════════════════════════════════════════════════════════════════════════════

async function runBacktest() {
    showLoading('Calcul du backtest...');
    const cost = parseFloat(document.getElementById('transaction_cost').value);

    try {
        const resp = await fetch('/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ transaction_cost: cost })
        });
        const data = await resp.json();
        if (data.error) { alert('Erreur: ' + data.error); return; }

        const sharpeEl = document.getElementById('sharpeValue');
        sharpeEl.textContent = data.sharpe_ratio.toFixed(3);
        sharpeEl.className = 'kpi-value ' + (data.sharpe_ratio > data.sharpe_buyhold ? 'positive' : 'negative');
        document.getElementById('sharpeBH').textContent = `B&H: ${data.sharpe_buyhold.toFixed(3)}`;

        const pnlEl = document.getElementById('pnlValue');
        pnlEl.textContent = (data.total_pnl >= 0 ? '+' : '') + data.total_pnl.toFixed(2) + '%';
        pnlEl.className = 'kpi-value ' + (data.total_pnl >= data.total_pnl_buyhold ? 'positive' : 'negative');
        document.getElementById('pnlBH').textContent = `B&H: ${data.total_pnl_buyhold >= 0 ? '+' : ''}${data.total_pnl_buyhold.toFixed(2)}%`;

        document.getElementById('mddValue').textContent = data.max_drawdown.toFixed(2) + '%';
        document.getElementById('mddBH').textContent = `B&H: ${data.max_drawdown_buyhold.toFixed(2)}%`;
        document.getElementById('tradesCount').textContent = data.num_trades;

        if (data.period) {
            document.getElementById('testPeriod').style.display = 'flex';
            document.getElementById('testPeriodText').textContent = data.period;
        }

        Plotly.newPlot('portfolioChart', [
            { x: data.dates, y: data.portfolio_values, type: 'scatter', mode: 'lines', name: 'Strategy', line: { color: '#c9a84c', width: 1.5 }, fill: 'tozeroy', fillcolor: 'rgba(201, 168, 76, 0.07)' },
            { x: data.dates, y: data.buyhold_values, type: 'scatter', mode: 'lines', name: 'Buy & Hold', line: { color: '#5a7fa8', width: 1, dash: 'dot' } }
        ], { ...plotlyLayout, showlegend: true, legend: { x: 0.02, y: 0.98, font: { size: 11, color: '#8a8a8a' } } }, plotlyConfig);

        const months = Object.keys(data.monthly_pnl);
        const pnls = Object.values(data.monthly_pnl);
        Plotly.newPlot('monthlyChart', [{
            x: months, y: pnls, type: 'bar',
            marker: { color: pnls.map(v => v >= 0 ? '#4caf7d' : '#e05c5c') }
        }], { ...plotlyLayout, xaxis: { ...plotlyLayout.xaxis, tickangle: -45 } }, plotlyConfig);

    } catch (e) {
        alert('Erreur: ' + e.message);
    } finally {
        hideLoading();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIGNAL
// ═══════════════════════════════════════════════════════════════════════════════

async function loadSignal() {
    showLoading('Chargement du signal...');
    try {
        const resp = await fetch('/signal');
        const data = await resp.json();
        if (data.error) { alert('Erreur: ' + data.error); return; }

        const signalClass = data.signal_text.toLowerCase();
        document.getElementById('signalValue').textContent = data.signal_text;
        document.getElementById('signalValue').className = 'signal-value ' + signalClass;
        document.getElementById('signalConfidence').textContent = (data.confidence * 100).toFixed(1) + '%';
        document.getElementById('signalPrice').textContent = '$' + data.price;
        document.getElementById('signalRegime').innerHTML = `<span class="badge ${data.regime.toLowerCase()}">${data.regime}</span>`;
        document.getElementById('signalDate').textContent = data.date;
        document.getElementById('headerPrice').textContent = '$' + data.price;

    } catch (e) {
        alert('Erreur: ' + e.message);
    } finally {
        hideLoading();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MONTE CARLO
// ═══════════════════════════════════════════════════════════════════════════════

function updateMcExplanation() {
    const method = document.getElementById('mc_method').value;
    const box = document.getElementById('mcExplanation');
    
    if (method === 'garch') {
        box.innerHTML = `
            <h3>◊ Monte Carlo GARCH + ML (Le Plus Réaliste)</h3>
            <ul>
                <li><strong>Volatilité variable GARCH(1,1):</strong> σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}</li>
                <li><strong>Clusters de volatilité:</strong> Périodes calmes suivies de périodes agitées (stylized fact)</li>
                <li><strong>Vrai modèle ML:</strong> RandomForest avec calcul complet des features sur chaque trajectoire</li>
                <li><strong>Très lent:</strong> 50-300 simulations max (estimation GARCH + features)</li>
                <li><strong>Le plus réaliste:</strong> Capture les dynamiques réelles des marchés financiers</li>
            </ul>
        `;
    } else if (method === 'ml') {
        box.innerHTML = `
            <h3>◊ Monte Carlo avec Vrai Modèle ML</h3>
            <ul>
                <li><strong>Calcul complet des features:</strong> RSI, MACD, Bollinger, etc. sur chaque trajectoire simulée</li>
                <li><strong>Vrai modèle RandomForest:</strong> Utilise le modèle pré-entraîné pour générer les signaux</li>
                <li><strong>Plus réaliste:</strong> Teste vraiment si les patterns ML fonctionnent sur données simulées</li>
                <li><strong>Plus lent:</strong> ~10x plus lent (50-200 simulations recommandées)</li>
            </ul>
        `;
    } else {
        box.innerHTML = `
            <h3>◊ Monte Carlo Simplifié (Régime SMA)</h3>
            <ul>
                <li><strong>Signaux basés sur régime:</strong> LONG si prix > SMA, sinon CASH</li>
                <li><strong>Rapide:</strong> Pas de calcul de features, jusqu'à 2000 simulations</li>
                <li><strong>Limitation:</strong> Ne teste pas le vrai modèle ML, juste la logique de régime</li>
                <li><strong>Utile pour:</strong> Vue rapide de l'impact du régime de marché</li>
            </ul>
        `;
    }
}

async function runMonteCarlo() {
    const method = document.getElementById('mc_method').value;
    const horizon = document.getElementById('mc_horizon').value;
    const n_simulations = parseInt(document.getElementById('mc_simulations').value);
    const cost = parseFloat(document.getElementById('mc_cost').value);
    
    const endpoint = method === 'garch' ? '/montecarlo_garch' : (method === 'ml' ? '/montecarlo_ml' : '/montecarlo');
    const loadingMsg = method === 'garch' ? 'Monte Carlo GARCH en cours (très lent)...' : 
                      (method === 'ml' ? 'Monte Carlo ML en cours (peut prendre du temps)...' : 'Simulation Monte Carlo...');
    
    showLoading(loadingMsg);

    try {
        const resp = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ horizon, n_simulations, transaction_cost: cost })
        });
        const data = await resp.json();
        if (data.error) { alert('Erreur: ' + data.error); return; }

        // Afficher les infos GBM ou GARCH
        document.getElementById('mcInfo').style.display = 'flex';
        let infoText = `Prix: <strong>$${data.current_price}</strong> · `;
        
        // Afficher les paramètres selon la méthode
        if (data.garch_params) {
            infoText += `α: <strong>${data.garch_params.alpha}</strong> · ` +
                `β: <strong>${data.garch_params.beta}</strong> · ` +
                `Persist: <strong>${data.garch_params.persistence}</strong> · ` +
                `Vol moy: <strong>${data.garch_params.avg_volatility_annual}%</strong>/an · `;
        } else {
            infoText += `Drift: <strong>${data.gbm_params.mu_annual}%</strong>/an · ` +
                `Vol: <strong>${data.gbm_params.sigma_annual}%</strong>/an · `;
        }
        
        infoText += `${data.n_simulations} sims × ${data.n_days} jours`;
        
        // Ajouter les stats de trading si disponibles (version ML)
        if (data.trading_stats) {
            infoText += ` · Trades moy: <strong>${data.trading_stats.avg_trades}</strong>`;
        }
        
        // Ajouter la méthode
        const methodLabel = data.method === 'garch_ml_model' ? '📈 GARCH+ML' : 
                           (data.method === 'full_ml_model' ? '🤖 ML' : '📊 Simplifié');
        infoText += ` · <span style="color: var(--accent-gold);">${methodLabel}</span>`;
        
        document.getElementById('mcInfoText').innerHTML = infoText;

        // KPIs
        document.getElementById('mcKpis').style.display = 'grid';
        
        const probProfit = data.probabilities.profit_strategy;
        document.getElementById('mcProbProfit').textContent = probProfit + '%';
        document.getElementById('mcProbProfitBar').style.width = probProfit + '%';
        
        const probBeat = data.probabilities.beat_buyhold;
        document.getElementById('mcProbBeat').textContent = probBeat + '%';
        document.getElementById('mcProbBeat').className = 'kpi-value ' + (probBeat >= 50 ? 'positive' : 'negative');
        document.getElementById('mcProbBeatBar').style.width = probBeat + '%';
        
        const meanRet = data.statistics.strategy.mean;
        document.getElementById('mcMeanReturn').textContent = (meanRet >= 0 ? '+' : '') + meanRet + '%';
        document.getElementById('mcMeanReturn').className = 'kpi-value ' + (meanRet >= 0 ? 'positive' : 'negative');
        document.getElementById('mcMeanBH').textContent = 'B&H: ' + (data.statistics.buyhold.mean >= 0 ? '+' : '') + data.statistics.buyhold.mean + '%';
        
        document.getElementById('mcVaR').textContent = data.statistics.strategy.p5 + '%';
        
        document.getElementById('mcMedian').textContent = (data.statistics.strategy.median >= 0 ? '+' : '') + data.statistics.strategy.median + '%';
        document.getElementById('mcMedian').className = 'kpi-value ' + (data.statistics.strategy.median >= 0 ? 'positive' : 'negative');
        document.getElementById('mcMedianBH').textContent = 'B&H: ' + (data.statistics.buyhold.median >= 0 ? '+' : '') + data.statistics.buyhold.median + '%';
        
        document.getElementById('mcBestCase').textContent = '+' + data.statistics.strategy.p95 + '%';

        // Graphique des trajectoires
        const trajectoryTraces = data.trajectories.map((traj, i) => ({
            x: data.dates,
            y: traj,
            type: 'scatter',
            mode: 'lines',
            line: { color: 'rgba(201, 168, 76, 0.2)', width: 0.8 },
            showlegend: false,
            hoverinfo: 'skip'
        }));
        
        // Ajouter la ligne de départ
        trajectoryTraces.push({
            x: [data.dates[0], data.dates[0]],
            y: [data.current_price * 0.5, data.current_price * 2],
            type: 'scatter',
            mode: 'lines',
            line: { color: 'rgba(255,255,255,0.2)', width: 1, dash: 'dash' },
            showlegend: false
        });

        Plotly.newPlot('mcTrajectoriesChart', trajectoryTraces, {
            ...plotlyLayout,
            yaxis: { ...plotlyLayout.yaxis, title: { text: 'Prix ($)', font: { size: 10, color: '#666' } } },
            xaxis: { ...plotlyLayout.xaxis, title: { text: 'Date', font: { size: 10, color: '#666' } } }
        }, plotlyConfig);

        // Histogramme des rendements
        Plotly.newPlot('mcDistributionChart', [
            {
                x: data.strategy_returns,
                type: 'histogram',
                name: 'Stratégie',
                marker: { color: 'rgba(201, 168, 76, 0.6)' },
                nbinsx: 40
            },
            {
                x: data.buyhold_returns,
                type: 'histogram',
                name: 'Buy & Hold',
                marker: { color: 'rgba(90, 127, 168, 0.4)' },
                nbinsx: 40
            }
        ], {
            ...plotlyLayout,
            barmode: 'overlay',
            showlegend: true,
            legend: { x: 0.02, y: 0.98, font: { size: 11, color: '#8a8a8a' } },
            xaxis: { ...plotlyLayout.xaxis, title: { text: 'Rendement (%)', font: { size: 10, color: '#666' } } },
            yaxis: { ...plotlyLayout.yaxis, title: { text: 'Fréquence', font: { size: 10, color: '#666' } } }
        }, plotlyConfig);

        // Box plot comparaison
        Plotly.newPlot('mcComparisonChart', [
            {
                y: data.strategy_returns,
                type: 'box',
                name: 'Stratégie ML',
                marker: { color: '#c9a84c' },
                boxmean: true
            },
            {
                y: data.buyhold_returns,
                type: 'box',
                name: 'Buy & Hold',
                marker: { color: '#5a7fa8' },
                boxmean: true
            }
        ], {
            ...plotlyLayout,
            showlegend: false,
            yaxis: { ...plotlyLayout.yaxis, title: { text: 'Rendement (%)', font: { size: 10, color: '#666' } }, zeroline: true, zerolinecolor: 'rgba(255,255,255,0.1)' }
        }, plotlyConfig);

    } catch (e) {
        alert('Erreur: ' + e.message);
    } finally {
        hideLoading();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// INITIALISATION
// ═══════════════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    // Event listener pour le sélecteur de méthode Monte Carlo
    const methodSelect = document.getElementById('mc_method');
    if (methodSelect) {
        methodSelect.addEventListener('change', updateMcExplanation);
    }
    
    // Charger le prix initial
    fetch('/signal').then(r => r.json()).then(data => {
        if (data.price) document.getElementById('headerPrice').textContent = '$' + data.price;
    }).catch(() => {});
});
