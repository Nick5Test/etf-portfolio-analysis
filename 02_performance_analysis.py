#CAGR, Volatilità, Sharpe Ratio, Max Drawdown

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
 
RISK_FREE_RATE = 0.02   # 2% annuo
DATA_DIR       = "data"
TRADING_DAYS   = 252 

def load_prices(filepath: str) -> pd.DataFrame:
    """
    Carica il CSV salvato in download_and_quality.
    """        
    prices = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"✅ Dati caricati: {len(prices)} giorni, {len(prices.columns)} ETF")
    return prices

#rendimenti giornalieri
def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola i rendimenti giornalieri percentuali.
    """
    return prices.pct_change().dropna()

#cagr
def compute_cagr(prices: pd.DataFrame) -> pd.Series:
    """
    Compound Annual Growth Rate — rendimento annualizzato.
    Formula: (prezzo_finale / prezzo_iniziale) ^ (1 / anni) - 1
    """
    years         = len(prices) / TRADING_DAYS
    total_return  = prices.iloc[-1] / prices.iloc[0]
    cagr          = total_return ** (1 / years) - 1
    return cagr

def compute_volatility(daily_returns: pd.DataFrame) -> pd.Series:
    """
    Volatilità annualizzata — deviazione standard dei rendimenti.
    Moltiplichiamo per sqrt(252) per passare da giornaliera ad annuale.
    """
    return daily_returns.std() * np.sqrt(TRADING_DAYS)

def compute_sharpe(cagr: pd.Series, volatility: pd.Series) -> pd.Series:
    """
    Sharpe Ratio — rendimento aggiustato per il rischio.
    Formula: (CAGR - risk_free_rate) / volatilità
    Più è alto, meglio è.
    """
    return (cagr - RISK_FREE_RATE) / volatility

def compute_max_drawdown(prices: pd.DataFrame) -> pd.Series:
    """
    Perdita massima dal picco al minimo nella storia dell'ETF.
    Formula: (prezzo_minimo - picco_massimo) / picco_massimo
    """
    rolling_max   = prices.cummax()
    drawdown      = (prices - rolling_max) / rolling_max
    max_drawdown  = drawdown.min()
    return max_drawdown

def build_summary_table(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Costruisce un DataFrame con tutte le metriche per ogni ETF.
    """
    daily_returns = compute_daily_returns(prices)
    cagr          = compute_cagr(prices)
    volatility    = compute_volatility(daily_returns)
    sharpe        = compute_sharpe(cagr, volatility)
    max_drawdown  = compute_max_drawdown(prices)
 
    summary = pd.DataFrame({
        "Settore":        [ETFs[t] for t in prices.columns],
        "CAGR %":         (cagr * 100).round(2),
        "Volatilità %":   (volatility * 100).round(2),
        "Sharpe Ratio":   sharpe.round(3),
        "Max Drawdown %": (max_drawdown * 100).round(2),
    }, index=prices.columns)
 
    summary = summary.sort_values("Sharpe Ratio", ascending=False)
    return summary

def plot_cagr(summary: pd.DataFrame):
    """
    Bar chart del CAGR per ogni ETF, ordinato dal migliore al peggiore.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
 
    colors = ["green" if v > 0 else "red" for v in summary["CAGR %"]]
    bars   = ax.bar(summary.index, summary["CAGR %"], color=colors, alpha=0.8)
 
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_title("CAGR per ETF (rendimento annualizzato)", fontsize=14)
    ax.set_ylabel("CAGR %")
    ax.set_xlabel("ETF")
    ax.grid(axis="y", alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "02_cagr.png"), dpi=150)
    plt.show()

def plot_risk_return(summary: pd.DataFrame):
    """
    Scatter plot Rischio vs Rendimento.
    Ogni punto è un ETF — in alto a sinistra = ideale (alto rendimento, basso rischio).
    """
    fig, ax = plt.subplots(figsize=(10, 7))
 
    ax.scatter(
        summary["Volatilità %"],
        summary["CAGR %"],
        s=150,
        alpha=0.8,
        color="steelblue",
    )
 
    # Etichetta ogni punto col ticker
    for ticker, row in summary.iterrows():
        ax.annotate(
            ticker,
            (row["Volatilità %"], row["CAGR %"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
        )
 
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Rischio vs Rendimento per ETF", fontsize=14)
    ax.set_xlabel("Volatilità % (rischio)")
    ax.set_ylabel("CAGR % (rendimento)")
    ax.grid(alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "03_risk_return.png"), dpi=150)
    plt.show()

def plot_max_drawdown(summary: pd.DataFrame):
    """
    Bar chart del Max Drawdown — quanto ha perso ogni ETF nel momento peggiore.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
 
    ax.bar(summary.index, summary["Max Drawdown %"], color="crimson", alpha=0.8)
    ax.set_title("Max Drawdown per ETF (perdita massima storica)", fontsize=14)
    ax.set_ylabel("Max Drawdown %")
    ax.set_xlabel("ETF")
    ax.grid(axis="y", alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "04_max_drawdown.png"), dpi=150)
    plt.show()

if __name__ == "__main__":
 
    # Carica i dati puliti
    prices = load_prices(os.path.join(DATA_DIR, "prices_clean.csv"))
 
    #tabella con tutte le metriche
    summary = build_summary_table(prices)
 
    #riepilogo nel terminale
    print("\n📊 Performance Summary (ordinato per Sharpe Ratio):")
    print("=" * 70)
    print(summary.to_string())
    print("=" * 70)
 
    #Salva il summary come CSV
    summary.to_csv(os.path.join(DATA_DIR, "performance_summary.csv"))
    print("\n💾 Summary salvato in data/performance_summary.csv")
 
    #Grafici
    plot_cagr(summary)
    plot_risk_return(summary)
    plot_max_drawdown(summary)
 
    print("\n✅ Fase 2 completata! Proseguo con 03_crisis_analysis.py")