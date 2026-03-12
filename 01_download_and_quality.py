import yfinance as yf
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

START_DATE = "1998-01-01"
END_DATE   = "2024-12-31"
DATA_DIR   = "data"

os.makedirs(DATA_DIR, exist_ok=True)

def download_data(tickers: dict, start: str, end: str) -> pd.DataFrame:
    """
    Scarica i prezzi di chiusura adjusted per tutti i ticker.
    Ritorna un DataFrame con una colonna per ticker.
    """
    print("⬇️  Download dati in corso...")
    raw = yf.download(
        tickers=list(tickers.keys()),
        start=start,
        end=end,
        auto_adjust=True,   #prezzi aggiustati per split e dividendi
        progress=True,
    )
    #colonna delle chiusure
    prices = raw["Close"]
    prices.columns.name = None
    print(f"✅ Scaricati {len(prices)} giorni di dati per {len(prices.columns)} ETF\n")
    return prices

def data_quality_report(prices: pd.DataFrame, tickers: dict) -> pd.DataFrame:
    """
    Controlla la qualità del dataset e stampa un report dettagliato.
    Ritorna un DataFrame con il summary per ogni ETF.
    """
    print("🔍 Data Quality Report")
    print("=" * 60)

    report = []

    for ticker in prices.columns:
        series = prices[ticker]

        first_date = series.first_valid_index()
        last_date  = series.last_valid_index()
        null_count   = series.isnull().sum()
        #% di nulli
        null_pct     = round(null_count / len(series) * 100, 2)

        years        = (last_date - first_date).days / 365
        expected_days = int(years * 252) #i mercati aprono ~252 giorni/anno
        actual_days   = series.dropna().count()
        coverage_pct  = round(actual_days / expected_days * 100, 1)

        
        daily_returns = series.pct_change().dropna()
        extreme_moves = (daily_returns.abs() > 0.20).sum()  #bool verifica se ci sono fluttuazioni >20. somma i true 

        report.append({
            "Ticker":       ticker,
            "Settore":      tickers[ticker],
            "Inizio dati":  first_date.strftime("%Y-%m-%d"),
            "Fine dati":    last_date.strftime("%Y-%m-%d"),
            "Anni storia":  round(years, 1),
            "Valori null":  null_count,
            "Null %":       null_pct,
            "Copertura %":  coverage_pct,
            "Movimenti >20%": extreme_moves,
        })

        status = "✅" if null_pct == 0 and coverage_pct > 95 else "⚠️ "
        print(f"{status} {ticker:5s} ({tickers[ticker]:<25}) | "
              f"Inizio: {first_date.strftime('%Y-%m-%d')} | "
              f"Null: {null_count:3d} ({null_pct}%) | "
              f"Copertura: {coverage_pct}%")

    print("=" * 60)
    return pd.DataFrame(report)

def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Rimuove righe con troppi null e interpola i gap piccoli.
    """
    #teniamo solo le righe dove almeno il 70% degli ETF ha dati
    threshold = int(len(prices.columns) * 0.7)
    prices_clean = prices.dropna(thresh=threshold)

    # Interpolazione lineare per gap piccoli (max 3 giorni consecutivi)
    prices_clean = prices_clean.interpolate(method="linear", limit=3)

    # Forward fill residui (es. festività locali)
    prices_clean = prices_clean.ffill(limit=1)

    removed_rows = len(prices) - len(prices_clean)
    print(f"\n🧹 Pulizia completata:")
    print(f"   Righe rimosse:     {removed_rows}")
    print(f"   Righe rimanenti:   {len(prices_clean)}")
    print(f"   Null residui:      {prices_clean.isnull().sum().sum()}")

    return prices_clean

#salvataggio dati
def save_data(prices: pd.DataFrame, filename: str):
    path = os.path.join(DATA_DIR, filename)
    prices.to_csv(path)
    print(f"\n💾 Dati salvati in: {path}")
    
#Visualizzazione grafico
def plot_normalized_prices(prices: pd.DataFrame, tickers: dict):
    """
    Normalizza i prezzi a 100 alla data di inizio comune
    e plotta le performance comparative.
    """
    # Data di inizio comune (quando tutti gli ETF hanno dati)
    common_start = prices.dropna().index[0]
    prices_common = prices.loc[common_start:]
 
    # Normalizzazione a base 100
    normalized = (prices_common / prices_common.iloc[0]) * 100
 
    fig, ax = plt.subplots(figsize=(14, 7))
 
    colors = plt.cm.tab10(np.linspace(0, 1, len(normalized.columns)))
    for i, ticker in enumerate(normalized.columns):
        ax.plot(
            normalized.index,
            normalized[ticker],
            label=f"{ticker} – {tickers[ticker]}",
            linewidth=1.5,
            color=colors[i],
        )
 
    # Evidenzia le crisi principali
    crises = [
        ("2000-03-01", "2002-10-01", "Dot-com"),
        ("2008-09-01", "2009-03-01", "Financial Crisis"),
        ("2020-02-01", "2020-04-01", "COVID"),
        ("2022-01-01", "2022-10-01", "Rate Hikes"),
    ]
    for start, end, label in crises:
        ax.axvspan(start, end, alpha=0.1, color="red")
        ax.text(
            pd.Timestamp(start), ax.get_ylim()[1] * 0.95,
            label, fontsize=7, color="red", alpha=0.7,
        )
 
    ax.set_title("Performance normalizzata ETF settoriali (base 100)", fontsize=14)
    ax.set_ylabel("Valore (base 100)")
    ax.set_xlabel("Data")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "01_performance_normalizzata.png"), dpi=150)
    plt.show()
    print("📊 Grafico salvato.")


if __name__ == "__main__":
 
    # 1. Download
    prices_raw = download_data(ETFs, START_DATE, END_DATE)
 
    # 2. Quality check
    quality_df = data_quality_report(prices_raw, ETFs)
    print("\nSummary qualità dati:")
    print(quality_df.to_string(index=False))
 
    # 3. Pulizia
    prices_clean = clean_prices(prices_raw)
 
    # 4. Salvataggio
    save_data(prices_raw,   "prices_raw.csv")
    save_data(prices_clean, "prices_clean.csv")
 
    # 5. Plot
    plot_normalized_prices(prices_clean, ETFs)
    print("\n✅ Fase 1 completata! Prosegui con 02_performance_analysis.py")
