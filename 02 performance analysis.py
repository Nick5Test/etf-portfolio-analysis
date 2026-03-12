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