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

#inizio e fine del momento peggiore 
CRISES = {
    "Dot-com (2000-2002)":       ("2000-03-01", "2002-10-31"),
    "Financial Crisis (2008)":   ("2008-09-01", "2009-03-31"),
    "COVID (2020)":              ("2020-02-01", "2020-04-30"),
    "Rate Hikes (2022)":         ("2022-01-01", "2022-12-31"),
}
 
DATA_DIR = "data"

def load_prices(filepath: str) -> pd.DataFrame:
    prices = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"✅ Dati caricati: {len(prices)} giorni, {len(prices.columns)} ETF")
    return prices

def compute_crisis_drawdown(prices: pd.DataFrame, start: str, end: str) -> pd.Series:
    """
    Calcola il drawdown massimo di ogni ETF nel periodo di crisi.
    Prende il prezzo del giorno prima della crisi come baseline
    e misura la perdita massima durante il periodo.
    """
    pre_crisis_price = prices.loc[:start].iloc[-1] 
    crisis_prices = prices.loc[start:end]
    drawdown = (crisis_prices - pre_crisis_price) / pre_crisis_price
    return drawdown.min()

def compute_recovery_days(prices: pd.DataFrame, start: str, end: str) -> pd.Series:
    """
    Calcola quanti giorni ha impiegato ogni ETF a tornare
    ai livelli pre-crisi dopo il punto più basso.
    Restituisce NaN se l'ETF non ha ancora recuperato.
    """
    pre_crisis_price = prices.loc[:start].iloc[-1] 
    post_crisis_prices = prices.loc[start:]
    recovery_days = {}
 
    for ticker in prices.columns:
        recovered = post_crisis_prices[ticker] >= pre_crisis_price[ticker]
 
        if recovered.any():
            first_recovery = recovered.idxmax() 
            days = (first_recovery - pd.Timestamp(start)).days
            recovery_days[ticker] = days
        else:
            # Non ha ancora recuperato
            recovery_days[ticker] = np.nan
 
    return pd.Series(recovery_days)

def build_crisis_summary(prices: pd.DataFrame) -> dict:
    """
    Costruisce un dizionario con una tabella riepilogativa per ogni crisi.
    Chiave: nome della crisi
    Valore: DataFrame con drawdown e giorni di recupero per ogni ETF
    """
    crisis_summaries = {}
 
    for crisis_name, (start, end) in CRISES.items():
 
        # Verifica che gli ETF abbiano dati per questo periodo
        available = prices.loc[start:end].dropna(axis=1, how="all").columns
        prices_available = prices[available]
 
        drawdown     = compute_crisis_drawdown(prices_available, start, end)
        recovery     = compute_recovery_days(prices_available, start, end)
 
        summary = pd.DataFrame({
            "Settore":          [ETFs[t] for t in available],
            "Drawdown %":       (drawdown * 100).round(2),
            "Giorni recupero":  recovery,
        }, index=available)
 
        # Ordina dal meno colpito al più colpito
        summary = summary.sort_values("Drawdown %", ascending=False)
        crisis_summaries[crisis_name] = summary
 
    return crisis_summaries

def plot_crisis_drawdowns(crisis_summaries: dict):
    """
    4 grafici a barre — uno per ogni crisi.
    Ogni barra mostra il drawdown di un ETF in quella crisi.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
 
    axes_flat = axes.flatten()
 
    for i, (crisis_name, summary) in enumerate(crisis_summaries.items()):
        ax = axes_flat[i]
 
        colors = ["crimson" if v < -20 else "orange" if v < -10 else "steelblue"
                  for v in summary["Drawdown %"]]
 
        bars = ax.bar(summary.index, summary["Drawdown %"], color=colors, alpha=0.85)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_title(crisis_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Drawdown %")
        ax.set_xlabel("ETF")
        ax.grid(axis="y", alpha=0.3)
 
    plt.suptitle("Drawdown per ETF nelle crisi storiche", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "05_crisis_drawdowns.png"), dpi=150)
    plt.show()
    print("📊 Grafico drawdown crisi salvato.")
 
 
def plot_crisis_heatmap(crisis_summaries: dict):
    """
    Heatmap — righe = ETF, colonne = crisi.
    Colore più scuro = perdita maggiore.
    Ti permette di vedere a colpo d'occhio quali ETF sono
    più difensivi e quali più vulnerabili.
    """
    #matrice ETF x Crisi
    heatmap_data = pd.DataFrame({
        crisis_name: summary["Drawdown %"]
        for crisis_name, summary in crisis_summaries.items()
    })
 
    # Ordina per drawdown medio — i più difensivi in cima
    heatmap_data["Media"] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values("Media", ascending=False)
    heatmap_data = heatmap_data.drop(columns="Media")
 
    fig, ax = plt.subplots(figsize=(12, 7))
 
    sns.heatmap(
        heatmap_data,
        annot=True,          # mostra i valori nelle celle
        fmt=".1f",           # un decimale
        cmap="RdYlGn",       # rosso = negativo, giallo = neutro, verde = positivo
        center=0,            # il centro della scala è 0
        linewidths=0.5,
        ax=ax,
    )
 
    ax.set_title("Drawdown % per ETF nelle crisi storiche\n(verde = resistito meglio, rosso = colpito di più)",
                 fontsize=13)
    ax.set_ylabel("ETF")
    ax.set_xlabel("Crisi")
 
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "06_crisis_heatmap.png"), dpi=150)
    plt.show()
    print("📊 Heatmap crisi salvata.")
 
 
def plot_recovery_comparison(crisis_summaries: dict):
    """
    Confronta i giorni di recupero per ogni ETF nelle diverse crisi.
    NaN = non ha ancora recuperato — viene mostrato come barra tratteggiata.
    """
    recovery_data = pd.DataFrame({
        crisis_name: summary["Giorni recupero"]
        for crisis_name, summary in crisis_summaries.items()
    })
 
    max_days   = recovery_data.max().max()
    fill_value = max_days * 1.2
    recovery_filled = recovery_data.fillna(fill_value)
 
    fig, ax = plt.subplots(figsize=(14, 7))
 
    x      = np.arange(len(recovery_data.index))
    width  = 0.2
    colors = ["steelblue", "darkorange", "crimson", "green"]
 
    for i, (crisis_name, color) in enumerate(zip(recovery_data.columns, colors)):
        values  = recovery_filled[crisis_name]
        is_nan  = recovery_data[crisis_name].isna()
 
        bars = ax.bar(x + i * width, values, width, label=crisis_name,
                      color=color, alpha=0.7)
 
        # Etichetta le barre non recuperate con "N/R"
        for j, (bar, nan) in enumerate(zip(bars, is_nan)):
            if nan:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 10,
                    "N/R", ha="center", fontsize=7, color=color
                )
 
    ax.set_title("Giorni di recupero post-crisi per ETF\n(N/R = non recuperato)", fontsize=13)
    ax.set_ylabel("Giorni")
    ax.set_xlabel("ETF")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(recovery_data.index)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "07_recovery_days.png"), dpi=150)
    plt.show()
    print("📊 Grafico recupero salvato.")

if __name__ == "__main__":
 
    prices = load_prices(os.path.join(DATA_DIR, "prices_clean.csv"))
    crisis_summaries = build_crisis_summary(prices)
    for crisis_name, summary in crisis_summaries.items():
        print(f"\n📉 {crisis_name}")
        print("=" * 60)
        print(summary.to_string())
    for crisis_name, summary in crisis_summaries.items():
        filename = crisis_name.replace(" ", "_").replace("(", "").replace(")", "") + ".csv"
        summary.to_csv(os.path.join(DATA_DIR, filename))
    print("\n💾 Summary crisi salvati in data/")
 

    plot_crisis_drawdowns(crisis_summaries)
    plot_crisis_heatmap(crisis_summaries)
    plot_recovery_comparison(crisis_summaries)
 
    print("\n✅ Fase 3 completata!")