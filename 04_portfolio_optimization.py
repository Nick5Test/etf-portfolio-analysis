import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from pypfopt import EfficientFrontier, risk_models, expected_returns
warnings.filterwarnings("ignore")
 
ETFs = {
    "XLK":  "Technology",
    "XLE":  "Energy",
    "XLF":  "Financials",
    "XLV":  "Healthcare",
    "XLU":  "Utilities",
    "XLP":  "Consumer Staples",
    "XLY":  "Consumer Discretionary",
    "XLB":  "Materials",
    "GLD":  "Gold",
    "VNQ":  "Real Estate",
}
 
CRISES = {
    "Dot-com (2000-2002)":       ("2000-03-01", "2002-10-31"),
    "Financial Crisis (2008)":   ("2008-09-01", "2009-03-31"),
    "COVID (2020)":              ("2020-02-01", "2020-04-30"),
    "Rate Hikes (2022)":         ("2022-01-01", "2022-12-31"),
}
 
RISK_FREE_RATE = 0.02
DATA_DIR       = "data"
TRADING_DAYS   = 252
 
 
def load_prices(filepath: str) -> pd.DataFrame:
    prices = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"✅ Dati caricati: {len(prices)} giorni, {len(prices.columns)} ETF")
    return prices
 
 
def compute_optimization_inputs(prices: pd.DataFrame):
    """
    - mu: rendimenti attesi per ogni ETF
    - cov_matrix: matrice di covarianza tra gli ETF
    """
    # Rendimenti attesi — media storica annualizzata
    mu = expected_returns.mean_historical_return(prices)
 
    # Matrice di covarianza
    cov_matrix = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
 
    return mu, cov_matrix
 
 
def optimize_max_sharpe(mu: pd.Series, cov_matrix: pd.DataFrame) -> dict:
    """
    Trova il portafoglio con il massimo Sharpe Ratio —
    il miglior rapporto rendimento/rischio possibile.
    """
    ef = EfficientFrontier(mu, cov_matrix)
    ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
 
    #arrotonda i pesi piccoli a zero
    weights = ef.clean_weights()
 
    #rendimento atteso, volatilità e Sharpe
    performance = ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)
 
    return {"weights": weights, "performance": performance}
 
 
def optimize_min_volatility(mu: pd.Series, cov_matrix: pd.DataFrame) -> dict:
    """
    Trova il portafoglio con la minima volatilità possibile —
    il più difensivo in assoluto.
    """
    ef = EfficientFrontier(mu, cov_matrix)
    ef.min_volatility()
 
    weights     = ef.clean_weights()
    performance = ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)
 
    return {"weights": weights, "performance": performance}
 
 
def optimize_crisis_aware(prices: pd.DataFrame, mu: pd.Series,
                          cov_matrix: pd.DataFrame) -> dict:
   
    # Calcola drawdown medio nelle crisi per ogni ETF
    crisis_drawdowns = {}
    for crisis_name, (start, end) in CRISES.items():
        # Verifica disponibilità dati nel periodo
        available      = prices.loc[start:end].dropna(axis=1, how="all").columns
        crisis_prices  = prices[available].loc[start:end]
        pre_crisis     = prices[available].loc[:start].iloc[-1]
        drawdown       = ((crisis_prices - pre_crisis) / pre_crisis).min()
        crisis_drawdowns[crisis_name] = drawdown
 
    
    drawdown_df = pd.DataFrame(crisis_drawdowns).T
    mean_drawdown = drawdown_df.mean()
    resilience_score = -mean_drawdown
    resilience_score = resilience_score[resilience_score > 0]
    weights_raw = resilience_score / resilience_score.sum()
 
    # Arrotonda e converti in dizionario
    weights = weights_raw.round(4).to_dict()
 
    # Calcola la performance del portafoglio con questi pesi
    ef          = EfficientFrontier(mu, cov_matrix)
    weights_arr = np.array([weights.get(t, 0) for t in mu.index])
    weights_arr = weights_arr / weights_arr.sum()   # rinormalizza per sicurezza
 
    ret  = float(mu.values @ weights_arr)
    vol  = float(np.sqrt(weights_arr @ cov_matrix.values @ weights_arr))
    sharpe = (ret - RISK_FREE_RATE) / vol
 
    performance = (ret, vol, sharpe)
 
    return {"weights": weights, "performance": performance}
 
 
def print_portfolio_results(name: str, result: dict):
    """
    Stampa i pesi e la performance di un portafoglio in modo leggibile.
    """
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
 
    print("\n  Allocazione:")
    for ticker, weight in sorted(result["weights"].items(),
                                  key=lambda x: x[1], reverse=True):
        if weight > 0:
            bar   = "█" * int(weight * 50)   # barra proporzionale al peso
            label = ETFs.get(ticker, ticker)
            print(f"  {ticker:5s} {label:<25} {weight:6.1%}  {bar}")
 
    ret, vol, sharpe = result["performance"]
    print(f"\n  Rendimento atteso:  {ret:.2%}")
    print(f"  Volatilità:         {vol:.2%}")
    print(f"  Sharpe Ratio:       {sharpe:.3f}")
 
 
def plot_portfolio_comparison(results: dict):
    """
    Bar chart che confronta i 3 portafogli su 3 metriche:
    rendimento, volatilità e Sharpe Ratio.
    """
    portfolios = list(results.keys())
    metrics    = ["Rendimento %", "Volatilità %", "Sharpe Ratio"]
 
    # Estrae le performance in un DataFrame
    perf_data = {}
    for name, result in results.items():
        ret, vol, sharpe = result["performance"]
        perf_data[name] = {
            "Rendimento %": ret * 100,
            "Volatilità %": vol * 100,
            "Sharpe Ratio": sharpe,
        }
 
    perf_df = pd.DataFrame(perf_data).T
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
 
    colors = ["steelblue", "darkorange", "green"]
 
    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(portfolios, perf_df[metric], color=colors, alpha=0.8)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticklabels(portfolios, rotation=15, ha="right")
 
    plt.suptitle("Confronto tra portafogli ottimizzati", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "08_portfolio_comparison.png"), dpi=150)
    plt.show()
    print("📊 Grafico confronto portafogli salvato.")
 
 
def plot_allocations(results: dict):
    """
    3 pie chart — uno per ogni portafoglio.
    Mostra visivamente come il capitale viene distribuito tra gli ETF.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
 
    for i, (name, result) in enumerate(results.items()):
        ax      = axes[i]
        weights = {k: v for k, v in result["weights"].items() if v > 0.01}
        labels  = [f"{t}\n{ETFs.get(t, t)}" for t in weights.keys()]
        values  = list(weights.values())
 
        ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",    # mostra la percentuale su ogni fetta
            startangle=90,        # inizia dal punto più alto
            pctdistance=0.85,     # distanza del testo dal centro
        )
        ax.set_title(name, fontsize=11, fontweight="bold")
 
    plt.suptitle("Allocazione dei portafogli ottimizzati", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "09_portfolio_allocations.png"), dpi=150)
    plt.show()
    print("📊 Grafico allocazioni salvato.")
 
 
def plot_correlation_heatmap(prices: pd.DataFrame):
    """
    Mostra la correlazione tra gli ETF.
    Correlazione alta = si muovono insieme = meno diversificazione.
    Correlazione bassa = si muovono indipendentemente = più diversificazione.
    Un buon portafoglio mescola ETF con bassa correlazione tra loro.
    """
    daily_returns = prices.pct_change().dropna()
    corr_matrix   = daily_returns.corr()
 
    fig, ax = plt.subplots(figsize=(10, 8))
 
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,      # minimo della scala colori
        vmax=1,       # massimo della scala colori
        linewidths=0.5,
        ax=ax,
    )
 
    ax.set_title("Matrice di correlazione tra ETF\n"
                 "(verde = bassa correlazione = buona diversificazione)",
                 fontsize=12)
 
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "10_correlation_heatmap.png"), dpi=150)
    plt.show()
    print("📊 Heatmap correlazione salvata.")
 
 
if __name__ == "__main__":
 
    prices = load_prices(os.path.join(DATA_DIR, "prices_clean.csv"))
 
    print("\n⚙️  Calcolo input per ottimizzazione...")
    mu, cov_matrix = compute_optimization_inputs(prices)
 
    print("🔧 Ottimizzazione in corso...")
    results = {
        "Max Sharpe":    optimize_max_sharpe(mu, cov_matrix),
        "Min Volatilità": optimize_min_volatility(mu, cov_matrix),
        "Crisis-Aware":  optimize_crisis_aware(prices, mu, cov_matrix),
    }
 
    for name, result in results.items():
        print_portfolio_results(name, result)
 
    for name, result in results.items():
        weights_df = pd.DataFrame.from_dict(
            result["weights"], orient="index", columns=["Peso"]
        )
        weights_df["Settore"] = [ETFs.get(t, t) for t in weights_df.index]
        filename = name.replace(" ", "_") + "_weights.csv"
        weights_df.to_csv(os.path.join(DATA_DIR, filename))
    print("\n💾 Pesi portafogli salvati in data/")
 
    plot_correlation_heatmap(prices)
    plot_portfolio_comparison(results)
    plot_allocations(results)
 
    print("\n✅ Fase 4 completata! Progetto terminato.")
    print("📁 Tutti i grafici e i CSV sono nella cartella data/")
