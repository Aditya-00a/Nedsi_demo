<div align="center">

# AI-Powered Stablecoin Risk Monitoring

### Ensemble Machine Learning + LLM Explainability

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aditya-00a/Nedsi_demo/blob/main/USDT_Monitoring.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Conference](https://img.shields.io/badge/NEDSI-2026-orange)](https://nedsi.org)

---

**Real-time fraud detection for USDT, USDC, DAI & BUSD using a weighted ensemble of Isolation Forest, One-Class SVM, and XGBoost — with Llama 3.3 70B generating SR 11-7 compliant explanations.**

[Launch Demo](#-quick-start) | [Architecture](#-system-architecture) | [Results](#-results) | [Paper](#-paper)

---

</div>

## Overview

Stablecoin fraud poses significant risk to the **$184 billion** cryptocurrency ecosystem. This system monitors live blockchain transactions and flags anomalies using three complementary ML approaches, then explains each decision in plain English for AML/CFT compliance.

<div align="center">

| Metric | Value |
|:------:|:-----:|
| **AUC-ROC** | 0.94 |
| **F1-Score** | 0.90 |
| **False Positive Rate** | 6% |
| **LLM Latency** | <500ms |
| **SR 11-7 Mapping** | 91% |

</div>

---

## Quick Start

### Option 1: Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aditya-00a/Nedsi_demo/blob/main/USDT_Monitoring.ipynb)

Click the badge above, set your API keys in Cell 2, and run all cells.

### Option 2: Run Locally

```bash
git clone https://github.com/Aditya-00a/Nedsi_demo.git
cd Nedsi_demo

# Set your API keys
export ETHERSCAN_API_KEY="your-key"
export GROQ_API_KEY="your-key"
export FRED_API_KEY="your-key"

# Launch
jupyter notebook USDT_Monitoring.ipynb
```

### API Keys Needed (all free)

| Service | Purpose | Get Key |
|---------|---------|---------|
| **Etherscan** | Live blockchain data | [etherscan.io/apis](https://etherscan.io/apis) |
| **Groq** | LLM inference (Llama 3.3 70B) | [console.groq.com](https://console.groq.com) |
| **FRED** | Macro data (Fed Funds, VIX) | [fred.stlouisfed.org/docs/api](https://fred.stlouisfed.org/docs/api/api_key.html) |

---

## System Architecture

```
                    +------------------+
                    |   Etherscan API   |  Live Ethereum transactions
                    +--------+---------+
                             |
                    +--------v---------+
                    | Feature Engineering|  22 liquidity-driven features
                    |  - Mint/Burn Ratio |  - Whale Concentration
                    |  - Exchange Flows  |  - Fed Funds Rate / VIX
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v---+  +------v------+  +----v-------+
     | Isolation   |  | One-Class   |  |  XGBoost   |
     | Forest      |  | SVM         |  | (Synthetic |
     | (35%)       |  | (25%)       |  |  Labels)   |
     +--------+---+  +------+------+  +----+-------+
              |              |              |     (40%)
              +--------------+--------------+
                             |
                    +--------v---------+
                    | Weighted Ensemble |  Risk Score 0-100%
                    +--------+---------+
                             |
                    +--------v---------+
                    |  Llama 3.3 70B   |  SR 11-7 compliant
                    |  (Groq - Free)   |  natural language
                    +--------+---------+  explanations
                             |
                    +--------v---------+
                    | Gradio Dashboard  |  Interactive UI
                    +------------------+
```

---

## Features

### Liquidity-Driven Feature Engineering

| Category | Features | Paper Reference |
|----------|----------|-----------------|
| **Liquidity Risk** | Mint-to-burn ratio, whale concentration, exchange flow imbalance | Section 3.3 |
| **Value Analysis** | Log value, value buckets, whale/dust flags | Section 3.3 |
| **Temporal** | Hour, day of week, business hours, night trading | Section 3.3 |
| **Behavioral** | Round amounts, sender frequency, exchange address detection | Section 3.3 |
| **Macroeconomic** | Federal Funds Rate, VIX (via FRED API) | Section 3.3 |
| **Network** | Gas analysis, address flow tracking | Section 3.3 |

### Supported Stablecoins

| Token | Market Cap | Contract |
|-------|-----------|----------|
| **USDT** (Tether) | $140B | `0xdac17f...` |
| **USDC** (USD Coin) | $38B | `0xa0b869...` |
| **DAI** | $4B | `0x6b1754...` |
| **BUSD** (Binance USD) | $2B | `0x4fabb1...` |

---

## Results

<div align="center">

| Model | AUC-ROC | F1 | Precision | Recall | FPR |
|:-----:|:-------:|:--:|:---------:|:------:|:---:|
| Isolation Forest | 0.87 | 0.82 | 0.80 | 0.84 | 12% |
| One-Class SVM | 0.81 | 0.76 | 0.74 | 0.78 | 18% |
| XGBoost | 0.91 | 0.87 | 0.86 | 0.88 | 8% |
| **Ensemble** | **0.94** | **0.90** | **0.89** | **0.91** | **6%** |

</div>

### Top Feature Importance (SHAP)

```
Whale Concentration  ██████████████████ 18%
Mint-to-Burn Ratio   ███████████████    15%
Exchange Net Flows   ████████████       12%
24h Volatility       ████████████       12%
Transaction Value    ██████████         10%
```

---

## Dashboard

The interactive Gradio dashboard provides:

- **Dashboard Tab** — Batch analysis with risk distribution and timeline charts
- **Transaction Analysis** — Deep-dive into individual transactions with risk gauge, model comparison, and LLM-generated compliance explanation
- **System Info** — Architecture overview and paper metrics

---

## Paper

> **AI-Powered Stablecoin Risk Monitoring Using Ensemble ML and LLM Explainability**
>
> *NEDSI 2026 Conference*

**Abstract:** This paper presents an AI-powered fraud detection system for stablecoins combining ensemble machine learning with Large Language Model explainability. The system monitors USDT, USDC, DAI, and BUSD using a weighted ensemble of Isolation Forest, One-Class SVM, and XGBoost trained on liquidity-driven features. Llama 3.3 70B generates regulatory-compliant explanations mapped to SR 11-7 model risk and AML/CFT frameworks.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **ML Models** | scikit-learn, XGBoost |
| **LLM** | Llama 3.3 70B via Groq |
| **Data** | Etherscan API, FRED API |
| **Dashboard** | Gradio, Plotly |
| **Explainability** | SHAP |

---

## Author

**Aditya Sakhale**

M.S. Management & Analytics (Risk Analytics) — NYU School of Professional Studies

[![GitHub](https://img.shields.io/badge/GitHub-Aditya--00a-181717?logo=github)](https://github.com/Aditya-00a)

---

<div align="center">

*Built for NEDSI 2026 — Northeast Decision Sciences Institute Annual Conference*

</div>
