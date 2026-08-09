# Cryptocurrency Market Analysis Dashboard

Measurement and visualization of cryptocurrency adoption, network activity, and
financial performance using on-chain and market data (Bitcoin & Ethereum).

**Live dashboard**: https://cryptocurrency-market-analysis-dashboard-mnerhxh2wpkuf8wgwdtk4.streamlit.app/

## Overview

This project analyzes 8,952 daily observations of Bitcoin and Ethereum on-chain
and market data across 15 features, addressing six research questions on how
adoption, network activity, and market behavior interact — building on a
50-paper systematic literature review (PRISMA 2020 methodology, screened down
from 480 initial records).

## Research Questions & Key Results

| RQ | Question | Best Model | Result |
|---|---|---|---|
| RQ1 | Do active addresses/wallets predict returns? | Linear Regression | R² = -3.29 (not effective for return modeling) |
| RQ2 | Which metrics predict returns vs. volatility? | Logistic Regression (volatility) | 96.81% accuracy, 100% recall |
| RQ3 | Genuine usage vs. speculative behavior? | LightGBM + SMOTE | 93.48% accuracy, 60.99% F1 |
| RQ4 | Do fees/hash rate predict adoption & value? | Random Forest | 95.20% accuracy (DAA), 93.63% (Market Cap) |
| RQ5 | Do liquidity/market cap detect market cycles? | Hidden Markov Model | Weak generalization (Test F1 = 0.000) |
| RQ6 | Does NVT ratio identify over/undervalued assets? | Logistic Regression | 54.6% accuracy — not reliable alone |

**Headline finding**: on-chain metrics are strong predictors of volatility and
adoption phase, but weak/unreliable for short-term return direction and
NVT-based valuation — on-chain data explains network health far better than
it explains price.

## Dataset

- **Source**: Blockchain.com, CoinMetrics, CoinGecko
- **Scope**: 8,952 daily observations, Bitcoin + Ethereum, 15 features (price,
  volume, market cap, transaction count/volume, daily active addresses, NVT,
  hash rate, transaction fees, volatility, liquidity)
- **Split**: 70% train (6,265) / 20% test (1,790) / 10% validation (896)

## Systematic Literature Review

Full review: [`docs/SLR.pdf`](docs/SLR.pdf)

- PRISMA 2020 methodology, 480 records screened down to 50 included studies
- Databases: Google Scholar, IEEE Xplore, ScienceDirect, SpringerLink, Wiley,
  INFORMS, SSRN (2021–2026, Q1 journals / CORE A/A* conferences only)
- Identified that RQ2 (returns/volatility) and RQ5 (best on-chain predictors)
  are the most comprehensively addressed in the literature, while RQ1
  (adoption → growth) and RQ4 (consensus participation → long-term value)
  remain under-addressed gaps

## Repository Structure

```
├── README.md
├── app/
│   ├── app.py              Streamlit dashboard
│   └── requirements.txt
├── data/
│   ├── crypto_dataset_2012_2026.csv
│   ├── train_70pct.csv
│   ├── test_20pct.csv
│   └── val_10pct.csv
├── docs/
│   ├── SLR.pdf              Systematic literature review
│   └── final_report_IEEE.pdf   IEEE-format final report
└── notebooks/
    └── analysis.ipynb       Full modeling pipeline (Colab)
```

## Tech Stack

Python (pandas, scikit-learn, XGBoost, LightGBM, hmmlearn) · Streamlit · matplotlib/SHAP

## Running the Dashboard Locally

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

## Authors

- **Maya KC** — Computer Science, Asian University for Women
- **Marzia Hassani**
- **Sonam Tsokid Lama**
- **Saraswoti Adhikari**
- **Noushin Subah**

Department of Computer Science, Asian University for Women, Chittagong, Bangladesh
