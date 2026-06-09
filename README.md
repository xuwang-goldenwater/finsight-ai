# 📊 FinSight AI — Value Investing Intelligence Platform
> **价值投资 AI 深度透视平台** · Powered by yfinance + OpenAI GPT-4o-mini

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?logo=streamlit)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What is FinSight AI?

FinSight AI is a bilingual (中文 / English) investment research platform that combines **quantitative DCF valuation** with a **3-layer LLM agent pipeline** to produce institutional-quality investment reports — all from a clean Streamlit dashboard.

It also includes a unique **Historical Blind-Test (Backtester)** module: simulate the AI making a BUY / HOLD / SELL decision at any past year-end, then verify the call against real price performance vs the S&P 500.

---

## ✨ Key Features

| Module | Description |
|--------|-------------|
| 📐 **Quantitative Engine** | ROE · ROIC · Gross Margin · FCF · D/E from yfinance; DCF intrinsic value with Margin of Safety |
| 🤖 **3-Layer Agent Pipeline** | Analyst → BrokerIntel → Investment Committee; each layer feeds the next |
| 🕰️ **Historical Blind-Test** | Time-slice financials to any past year, re-run full analysis, compare AI return vs SPY alpha |
| 🌐 **Bilingual Dashboard** | Full 简体中文 / English toggle — UI labels, LLM output, and report all switch language |
| 📄 **Exportable Reports** | Download full Markdown research reports locally |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                     app.py                          │
│              Streamlit Bilingual Dashboard          │
└──────────────┬──────────────────┬───────────────────┘
               │                  │
       ┌───────▼───────┐  ┌───────▼────────┐
       │financial_engine│  │  backtester.py │
       │   DCF + KPIs   │  │ Time-slice +   │
       │   (yfinance)   │  │ SPY Alpha Test │
       └───────┬────────┘  └───────┬────────┘
               │                  │
       ┌───────▼──────────────────▼────────┐
       │          analysis_agents.py        │
       │                                    │
       │  Layer 1: FinancialAnalystAgent    │
       │  Layer 2: BrokerIntelAgent         │
       │  Layer 3: InvestmentCommitteeAgent │
       └────────────────────────────────────┘
```

### Agent Pipeline

```
Financial Data (yfinance)
        │
        ▼
Layer 1 │ FinancialAnalystAgent
        │ Causal analysis of ROE / ROIC / FCF / DCF
        ▼
Layer 2 │ BrokerIntelAgent
        │ Bull case · Bear case · Market consensus
        ▼
Layer 3 │ InvestmentCommitteeAgent
        │ Synthesizes all signals → Final report + BUY / HOLD / SELL
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/xuwang-goldenwater/finsight-ai.git
cd finsight-ai

pip install -r requirements.txt
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY="sk-..."
# or create a .env file:
echo "OPENAI_API_KEY=sk-..." > .env
```

### 3. Launch the Dashboard

```bash
streamlit run app.py
```

### 4. CLI Usage (without UI)

```bash
# Real-time analysis
python financial_engine.py AAPL
python analysis_agents.py NVDA

# Historical blind-test
python backtester.py AAPL --year 2022
python backtester.py MSFT --year 2021 --model gpt-4o
```

---

## 📦 Installation

### Requirements

```txt
streamlit>=1.35
yfinance>=0.2.50
openai>=1.0
pandas>=2.0
numpy>=1.24
python-dotenv
curl_cffi          # optional but recommended — bypasses Yahoo Finance rate limits
```

Install all at once:

```bash
pip install streamlit yfinance openai pandas numpy python-dotenv curl_cffi
```

---

## 📁 Project Structure

```
finsight-ai/
├── app.py                 # Streamlit bilingual dashboard (main entry)
├── financial_engine.py    # DCF valuation engine + 5 KPI metrics
├── analysis_agents.py     # 3-layer LLM agent pipeline
├── backtester.py          # Historical blind-test module
├── .gitignore
└── README.md
```

---

## 📊 Dashboard Walkthrough

### Live Analysis Mode (`📡`)

| Section | Content |
|---------|---------|
| 🏢 Company Card | Name · Price · Market Cap · 52-Week Range · IPO Date · Sector |
| 📐 Quant Dashboard | Latest ROE / ROIC / Gross Margin / D/E / FCF with color-coded strength signals |
| 🎯 Margin of Safety | Visual progress bar: current price vs AI intrinsic value |
| 🤖 Agent Report | Tab 1: Bull case · Tab 2: Bear case · Tab 3: Full committee report |

### Backtest Mode (`🕰️`)

| Section | Content |
|---------|---------|
| Decision Banner | 🟢 BUY · 🟡 HOLD · 🔴 SELL — the AI's verdict at year-end snapshot |
| Historical KPIs | Metrics sliced to the backtest year |
| Return Comparison | Entry price → Today · Ticker vs SPY (S&P 500) · Alpha |
| Decision Verdict | Was the AI right? Validated against actual price movement |
| Agent Reasoning | Bull / Bear / Macro context · Full historical committee report |

---

## 💰 Cost Estimate

Using `gpt-4o-mini` (default model):

| Operation | Tokens | Cost |
|-----------|--------|------|
| Full live analysis | ~5,000 | ~$0.001 |
| Historical backtest | ~8,000 | ~$0.0015 |
| 100 runs total | ~800K | ~$0.15 |

**Bottom line: extremely low cost for personal research use.**

---

## ⚙️ Configuration

Key parameters in `financial_engine.py`:

```python
DEFAULT_DISCOUNT_RATE   = 0.09   # WACC
DEFAULT_TERMINAL_GROWTH = 0.025  # Perpetual growth rate
DEFAULT_FCF_CAP         = 0.15   # FCF growth cap
FORECAST_YEARS          = 10     # DCF forecast horizon
```

Override via CLI:
```bash
python financial_engine.py AAPL --discount-rate 0.10 --terminal-growth 0.03
```

---

## ⚠️ Disclaimer

This platform is for **educational and research purposes only**.
It does not constitute financial advice. Always do your own due diligence before making investment decisions.

Historical backtests are subject to survivorship bias and hindsight bias. Past AI performance does not guarantee future results.

---

## 📄 License

MIT License — free to use, modify, and distribute.
