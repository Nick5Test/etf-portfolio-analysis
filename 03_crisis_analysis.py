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