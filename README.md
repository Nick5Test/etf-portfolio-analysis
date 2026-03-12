# 📊 ETF Sector Portfolio Analysis (1998–2024)

Analisi storica delle performance di 10 ETF settoriali per costruire un portafoglio ottimizzato.

## Obiettivo
Capire quali settori hanno performato meglio negli ultimi 25 anni, come hanno reagito alle crisi, e costruire un portafoglio ideale basato su dati reali.

## ETF Analizzati
| Ticker | Settore |
|--------|---------|
| XLK | Technology |
| XLE | Energy |
| XLF | Financials |
| XLV | Healthcare |
| XLU | Utilities |
| XLP | Consumer Staples |
| XLY | Consumer Discretionary |
| XLB | Materials |
| GLD | Gold |
| VNQ | Real Estate |

## Struttura del Progetto
```
etf-portfolio-analysis/
│
├── data/                          # Dati CSV e grafici generati
├── notebooks/
│   ├── 01_download_and_quality.py  # Download dati + data quality checks
│   ├── 02_performance_analysis.py  # CAGR, volatilità, Sharpe Ratio, drawdown
│   ├── 03_crisis_analysis.py       # Analisi comportamento nelle crisi
│   └── 04_portfolio_optimization.py# Efficient Frontier + portafoglio ottimale
├── README.md
└── requirements.txt
```

## Come Iniziare
```bash
# Installa dipendenze
pip install -r requirements.txt

# Esegui in ordine
python notebooks/01_download_and_quality.py
python notebooks/02_performance_analysis.py
python notebooks/03_crisis_analysis.py
python notebooks/04_portfolio_optimization.py
```

## Crisi Analizzate
- **Dot-com crash**: 2000–2002
- **Financial Crisis**: 2008–2009
- **COVID crash**: Feb–Apr 2020
- **Rate Hikes**: 2022

## Tech Stack
- `yfinance` — dati storici da Yahoo Finance
- `pandas` / `numpy` — manipolazione e calcoli
- `matplotlib` / `seaborn` — visualizzazioni
- `PyPortfolioOpt` — ottimizzazione portafoglio (Efficient Frontier)
