# 📊 问巴菲特与达利欧 | Ask Buffett & Dalio

> **Bilingual Value Investing Intelligence Platform**
> Powered by yfinance · OpenAI GPT-4o-mini · Ray Dalio's Principles · Warren Buffett's Value Framework

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?logo=streamlit)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What Is This?

**Ask Buffett & Dalio** is a bilingual (简体中文 / English) institutional-grade investment research platform that combines:

- **Quantitative DCF valuation** grounded in Buffett-style free cash flow analysis
- **A 3-layer LLM agent pipeline** culminating in a Ray Dalio–inspired Investment Committee report
- **Interactive investor chat** — ask follow-up questions to the AI committee in real time
- **Historical Blind-Test (Backtester)** — replay any past year-end decision and validate it against actual returns

All from a clean, Bloomberg-Terminal-inspired light dashboard built on Streamlit.

---

## ✨ Key Features

| Module | Description |
|--------|-------------|
| 📐 **Quantitative Engine** | ROE · ROIC · Gross Margin · FCF · D/E sourced live from yfinance; Two-Stage DCF with Agent-calibrated growth rates |
| 🤖 **3-Layer Agent Pipeline** | FinancialAnalyst → BrokerIntel → Investment Committee; each layer feeds the next |
| 🧠 **Dalio 5-Step Committee** | Committee report follows Ray Dalio's *Principles* framework: Goals → Problems & Blindspots → Diagnose → Design → Decision |
| 💬 **Interactive Investor Chat** | Ask the AI committee anything about the stock — multi-turn, context-aware, bilingual |
| 🕰️ **Historical Blind-Test** | Slice financials to any past year-end, re-run full analysis, compare AI return vs S&P 500 alpha |
| 🌐 **Fully Bilingual** | Every UI label, agent prompt, and report output switches between 简体中文 and English |
| 📄 **Exportable Reports** | Download full Markdown research reports locally |

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────┐
│                         app.py                             │
│         问巴菲特与达利欧 · Streamlit Dashboard              │
└──────────────────┬───────────────────┬─────────────────────┘
                   │                   │
         ┌─────────▼────────┐ ┌────────▼────────┐
         │ financial_engine │ │  backtester.py  │
         │  Two-Stage DCF   │ │ Time-slice +    │
         │  + 5 KPI metrics │ │ SPY Alpha Test  │
         └─────────┬────────┘ └────────┬────────┘
                   │                   │
         ┌─────────▼───────────────────▼─────────┐
         │           analysis_agents.py           │
         │                                        │
         │  Layer 1 · FinancialAnalystAgent        │
         │  Layer 2 · BrokerIntelAgent             │
         │  Layer 3 · InvestmentCommitteeAgent     │
         │           (Dalio 5-Step Framework)      │
         │                                        │
         │  + answer_investor_question()           │
         │    Interactive multi-turn chat          │
         └────────────────────────────────────────┘
```

### Agent Pipeline

```
Financial Data (yfinance)
        │
        ▼
Layer 1 │ FinancialAnalystAgent
        │ Causal analysis of ROE / ROIC / FCF / DCF
        │ Outputs: AI-calibrated Stage 1 & Stage 2 growth rates
        ▼
Layer 2 │ BrokerIntelAgent
        │ Bull case · Bear case · Market consensus
        ▼
Layer 3 │ InvestmentCommitteeAgent  ← Dalio's Principles
        │ Step 1 · Goals
        │ Step 2 · Problems & Blindspots
        │ Step 3 · Root-Cause Diagnosis
        │ Step 4 · Design + Formulation
        │ Step 5 · Action → DECISION: BUY / HOLD / SELL
        ▼
Interactive Chat │ answer_investor_question()
                 │ Multi-turn Q&A with full report context
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/xuwang-goldenwater/finsight-ai.git
cd finsight-ai
pip install -r requirements.txt
```

### 2. Set API Key

```bash
export OPENAI_API_KEY="sk-..."
# or create a .env file:
echo "OPENAI_API_KEY=sk-..." > .env
```

### 3. Launch the Dashboard

```bash
streamlit run app.py
```

### 4. CLI Usage (no UI)

```bash
# Live analysis
python financial_engine.py AAPL
python analysis_agents.py NVDA

# Historical blind-test
python backtester.py AAPL --year 2022
python backtester.py MSFT --year 2021 --model gpt-4o
```

---

## 📦 Requirements

```
streamlit>=1.35
yfinance>=0.2.50
pandas>=2.0
numpy>=1.24
openai>=1.0
python-dotenv>=1.0
curl_cffi>=0.6        # optional but strongly recommended — bypasses Yahoo Finance rate limits
```

---

## 📁 Project Structure

```
finsight-ai/
├── app.py                 # Streamlit bilingual dashboard (main entry)
├── financial_engine.py    # Two-Stage DCF engine + 5 KPI metrics
├── analysis_agents.py     # 3-layer LLM agent pipeline + interactive chat
├── backtester.py          # Historical blind-test module
├── requirements.txt
└── README.md
```

---

## 📊 Dashboard Walkthrough

### Live Analysis Mode (`📡`)

| Section | Content |
|---------|---------|
| 🏢 **Company Card** | Name · Price · Market Cap · 52-Week Range · Sector |
| 📐 **Quant Dashboard** | ROE / ROIC / Gross Margin / D/E / FCF — color-coded signal strength |
| 🎯 **Margin of Safety** | Visual progress bar: current price vs AI intrinsic value |
| 📈 **Two-Stage DCF** | Agent-calibrated growth rates (Stage 1 yr 1–5, Stage 2 yr 6–10) |
| 🤖 **Agent Report** | Tab 1: Bull case · Tab 2: Bear case · Tab 3: Full Dalio committee report |
| 💬 **Investor Chat** | Ask follow-up questions — answers grounded in the full report context |

### Backtest Mode (`🕰️`)

| Section | Content |
|---------|---------|
| **Decision Banner** | 🟢 BUY · 🟡 HOLD · 🔴 SELL — AI verdict at the chosen year-end snapshot |
| **Historical KPIs** | Metrics time-sliced to the backtest year |
| **Return Comparison** | Entry price → Today · Ticker return vs SPY · Alpha |
| **Decision Verdict** | Was the AI right? Validated against actual price movement |
| **Agent Reasoning** | Bull / Bear / Macro context · Full historical committee report |

---

## 🧠 The Dalio 5-Step Investment Committee

The Layer 3 committee report follows Ray Dalio's decision-making framework from *Principles*:

1. **Goals** — Define the investment objective and time horizon
2. **Problems & Blindspots** — Surface risks the market may be mispricing
3. **Root-Cause Diagnosis** — Identify what is truly driving the business and its valuation
4. **Design + Formulation** — Construct the investment thesis with scenario weighting
5. **Action** — Deliver a clear `DECISION: BUY / HOLD / SELL` with conviction rationale

---

## 💰 Cost Estimate

Default model: `gpt-4o-mini`

| Operation | Approx Tokens | Approx Cost |
|-----------|--------------|-------------|
| Full live analysis | ~5,000 | ~$0.001 |
| Historical backtest | ~8,000 | ~$0.0015 |
| Interactive chat (per turn) | ~1,500 | ~$0.0003 |
| 100 full analyses | ~800K | ~$0.15 |

**Bottom line: extremely low cost for personal research use.**

---

## ⚙️ DCF Configuration

Key parameters in `financial_engine.py`:

```python
DEFAULT_DISCOUNT_RATE    = 0.09   # WACC
DEFAULT_TERMINAL_GROWTH  = 0.025  # Perpetual growth rate
DEFAULT_FCF_CAP          = 0.15   # FCF growth cap
FORECAST_YEARS           = 10     # DCF forecast horizon
```

The AI agent automatically calibrates **Stage 1** (yr 1–5) and **Stage 2** (yr 6–10) growth rates based on its analysis of the company's fundamentals — no manual tuning needed.

---

## ⚠️ Disclaimer

This platform is for **educational and research purposes only**.
It does not constitute financial advice. Always conduct your own due diligence before making investment decisions.

Historical backtests are subject to survivorship bias and hindsight bias. Past AI performance does not guarantee future results.

---

## 📄 License

MIT License — free to use, modify, and distribute.
