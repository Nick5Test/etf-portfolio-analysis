📊 ETF Sector Portfolio Analysis (1998–2024)
Analisi storica delle performance di 10 ETF settoriali per costruire un portafoglio ottimizzato.
Progetto realizzato per approfondire Python applicato alla finanza quantitativa.
Obiettivo
Capire quali settori hanno performato meglio negli ultimi 25 anni, come hanno reagito alle crisi storiche, e costruire un portafoglio ideale basato su dati reali tramite ottimizzazione quantitativa.
ETF Analizzati
TickerSettoreXLKTechnologyXLEEnergyXLFFinancialsXLVHealthcareXLUUtilitiesXLPConsumer StaplesXLYConsumer DiscretionaryXLBMaterialsGLDGoldVNQReal Estate
Struttura del Progetto
etf-portfolio-analysis/
│
├── data/                            # Dati CSV e grafici generati automaticamente
├── 01_download_and_quality.py       # Download dati + data quality checks
├── 02_performance_analysis.py       # CAGR, volatilità, Sharpe Ratio, drawdown
├── 03_crisis_analysis.py            # Analisi comportamento nelle crisi storiche
├── 04_portfolio_optimization.py     # Efficient Frontier + portafoglio ottimale
├── README.md
└── requirements.txt
Come Iniziare
bash# Crea e attiva il virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# Installa le dipendenze
pip install -r requirements.txt

# Esegui gli script in ordine
python 01_download_and_quality.pygit 
python 02_performance_analysis.py
python 03_crisis_analysis.py
python 04_portfolio_optimization.py
Pipeline di Analisi
Fase 1 — Download e Data Quality
Scarica i dati storici da Yahoo Finance tramite yfinance. Esegue data quality checks su ogni ETF — valori nulli, copertura temporale, movimenti anomali. Produce prices_clean.csv usato da tutte le fasi successive.
Fase 2 — Analisi delle Performance
Calcola le metriche principali per ogni ETF: CAGR (rendimento annualizzato), volatilità, Sharpe Ratio e Max Drawdown. Produce grafici comparativi e un summary CSV ordinato per Sharpe Ratio.
Fase 3 — Analisi delle Crisi
Isola i 4 periodi di crisi e misura drawdown e velocità di recupero di ogni ETF. Produce una heatmap che mostra a colpo d'occhio quali ETF sono più difensivi storicamente.
Fase 4 — Ottimizzazione del Portafoglio
Costruisce 3 portafogli ottimizzati tramite Efficient Frontier:

Max Sharpe — massimizza il rapporto rendimento/rischio
Min Volatilità — minimizza il rischio complessivo
Crisis-Aware — pesa gli ETF in base alla resilienza nelle crisi storiche

Crisi Analizzate
CrisiPeriodoDot-com crash2000–2002Financial Crisis2008–2009COVID crashFeb–Apr 2020Rate Hikes2022
Risultati Principali
La matrice di correlazione mostra che GLD è l'asset più diversificante del portafoglio — correlazione vicina a zero con tutti gli altri ETF. Tra i portafogli ottimizzati, Max Sharpe ottiene il miglior rapporto rendimento/rischio (Sharpe 0.58), mentre Min Volatilità è il più difensivo nel quotidiano (volatilità 10.73%).
Limitazioni
I risultati sono basati su dati storici — le performance passate non garantiscono quelle future. L'ottimizzazione su 25 anni di storia potrebbe non produrre il portafoglio ottimale per i prossimi 25 anni. I pesi del portafoglio andrebbero ribilanciati periodicamente.
Tech Stack

yfinance — dati storici da Yahoo Finance
pandas / numpy — manipolazione e calcoli
matplotlib / seaborn — visualizzazioni
PyPortfolioOpt — ottimizzazione portafoglio (Efficient Frontier)
scikit-learn — dipendenza per la stima della covarianza (Ledoit-Wolf)