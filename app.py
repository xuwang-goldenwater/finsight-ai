"""
app.py
======
问巴菲特与达利欧 | Ask Buffett & Dalio
双语价值投资智能研究平台 · Bilingual Value Investing Intelligence Platform

Run:  streamlit run app.py
"""
from dotenv import load_dotenv; load_dotenv()
import re
import json
import math
import warnings
from datetime import date, datetime, timezone

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# 页面配置（必须第一个 st 调用）
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="问巴菲特与达利欧 | Ask Buffett & Dalio",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# 全局 CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ═══════════════════════════════════════════════════════════════════
   问巴菲特与达利欧 | Ask Buffett & Dalio
   Design System v4 — Light Tech Terminal
   Rule: CSS vars only, no hardcoded hex outside :root
   ═══════════════════════════════════════════════════════════════════ */

@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

:root {
    --bg-page:       #f8f9fa;
    --bg-card:       #ffffff;
    --bg-subtle:     #f3f4f6;
    --bg-overlay:    #e9ecef;
    --border:        #e1e4e8;
    --border-strong: #c9cdd2;
    --text-primary:  #1b1f23;
    --text-secondary:#57606a;
    --text-muted:    #8c959f;
    --accent:        #0550ae;
    --signal-bull:   #1a7f37;
    --signal-bear:   #c01c2c;
    --signal-hold:   #7d4e00;
    --mono:          'JetBrains Mono', 'Fira Code', 'SF Mono', ui-monospace, monospace;
}

/* ── Shell ──────────────────────────────────────────────────────── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] > div {
    background-color: var(--bg-page) !important;
}

/* ── Streamlit 顶部 Header 处理策略 ────────────────────────────────
   ⚠️  绝对不能对 stHeader 设置 height:0 / visibility:hidden ——
       stSidebarCollapsedControl（收起后的展开箭头）是 stHeader 的
       子元素，父级被杀死后展开按钮会永久消失、无法点击。
   正确做法：保留 header 的 DOM 与高度，只把视觉噪音逐项隐藏。
   ─────────────────────────────────────────────────────────────── */

/* 1. Header 壳体：背景透明、无边框，不影响布局高度 */
[data-testid="stHeader"],
header[data-testid="stHeader"] {
    background-color: transparent !important;
    border-bottom: none !important;
    box-shadow: none !important;
}

/* 2. 逐项隐藏不需要的子元素（display:none 彻底移出流） */
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stToolbarActions"] { display: none !important; }
#MainMenu { display: none !important; }
footer    { display: none !important; }

/* 3. 侧边栏展开按钮 —— 显式激活，浅色模式清晰可见 */
[data-testid="stSidebarCollapsedControl"] {
    background-color: var(--bg-overlay) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 6px !important;
    opacity: 1 !important;
    visibility: visible !important;
    pointer-events: auto !important;
    z-index: 999991 !important;
}
[data-testid="stSidebarCollapsedControl"]:hover {
    background-color: var(--bg-card) !important;
    border-color: var(--accent) !important;
}
[data-testid="stSidebarCollapsedControl"] svg {
    fill: var(--text-primary) !important;
    color: var(--text-primary) !important;
}

[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}

/* Hide native sidebar header — it renders "keyboard_double_arrow_left"
   as visible text when our monospace override breaks Material Symbols. */
[data-testid="stSidebarHeader"] {
    display: none !important;
}

/* ── Main panel ─────────────────────────────────────────────────── */
/* Header 仍在 DOM 中占位（高度约 3rem），内容从 header 下方自然开始 */
[data-testid="stMainBlockContainer"] {
    padding-top: 1rem !important;
    max-width: 1280px;
}

/* ── Sidebar 内容顶部留白 ────────────────────────────────────────── */
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.5rem !important;
}

/* ── Sidebar custom brand logo ──────────────────────────────────── */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 0 14px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}
.sidebar-brand-icon { font-size: 1.4rem; line-height: 1; }
.sidebar-brand-text {
    font-family: var(--mono) !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    color: var(--text-primary) !important;
    line-height: 1.4;
}

/* ── Typography ─────────────────────────────────────────────────── */
/* NOTE: <span> is intentionally excluded here. Spans inherit from their
   parent div/button (which does carry the monospace override), so our
   UI text stays monospace. But Material Symbols icon spans — which
   carry their own class-level font-family rule — can now correctly
   override the inherited value and render as glyphs, not raw text.   */
html, body,
p, li, div, label, td, th, button,
.stMarkdown, .stText, .stCaption, [class*="css"] {
    font-family: var(--mono) !important;
    -webkit-font-smoothing: antialiased;
}
html, body {
    background-color: var(--bg-page) !important;
    color: var(--text-primary) !important;
}
p, li { line-height: 1.65 !important; color: var(--text-primary); }
h1, h2, h3, h4 {
    font-family: var(--mono) !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.015em !important;
    line-height: 1.3 !important;
}
h1 { font-size: 1.4rem !important; }
h2 { font-size: 1.1rem !important; }
h3 { font-size: 0.95rem !important; }
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 18px 0 !important; }

/* ── Decision Badges ────────────────────────────────────────────── */
.badge-buy, .badge-hold, .badge-sell {
    display: inline-block;
    font-family: var(--mono);
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    padding: 5px 22px;
    border-radius: 4px;
    border: 1.5px solid;
    background: transparent;
}
.badge-buy  { color: var(--signal-bull); border-color: var(--signal-bull); }
.badge-hold { color: var(--signal-hold); border-color: var(--signal-hold); }
.badge-sell { color: var(--signal-bear); border-color: var(--signal-bear); }

/* ── Section Label ──────────────────────────────────────────────── */
.section-title {
    font-family: var(--mono);
    font-size: 0.63rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 0 0 0 9px;
    border-left: 2px solid var(--accent);
    margin: 20px 0 8px;
    line-height: 1.4;
    display: block;
}

/* ── KPI Box ────────────────────────────────────────────────────── */
/* Uses display:block spans — no overflow, no fixed height            */
.kpi-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 10px 14px 11px;
    margin-bottom: 8px;
}
.kpi-label {
    display: block;
    font-family: var(--mono);
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 3px;
    white-space: nowrap;
}
.kpi-value {
    display: block;
    font-family: var(--mono);
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    white-space: nowrap;
    line-height: 1.3;
}

/* ── Price Card (used inside st.columns — no flex needed here) ─── */
.price-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 13px 16px;
}
.price-card-label {
    display: block;
    font-family: var(--mono);
    font-size: 0.60rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 5px;
    white-space: nowrap;
}
.price-card-value {
    display: block;
    font-family: var(--mono);
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--text-primary);
    white-space: nowrap;
    line-height: 1.15;
}
.price-card-value.intrinsic { color: var(--accent); }

/* ── MoS percentage row ─────────────────────────────────────────── */
.mos-label {
    display: block;
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-secondary);
    margin-bottom: 4px;
}

/* ── Company Card ───────────────────────────────────────────────── */
.company-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 18px 22px 16px;
    margin-bottom: 20px;
}
.company-name {
    font-family: var(--mono);
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 2px;
    letter-spacing: -0.01em;
}
.company-meta {
    font-family: var(--mono);
    font-size: 0.73rem;
    color: var(--text-muted);
}
.company-tag {
    display: inline-block;
    background: var(--bg-subtle);
    color: var(--text-secondary);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1px 8px;
    font-family: var(--mono);
    font-size: 0.67rem;
    letter-spacing: 0.04em;
    margin: 4px 4px 0 0;
}
.price-big {
    font-family: var(--mono);
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--signal-bull);
    white-space: nowrap;
    letter-spacing: -0.01em;
}

/* ── Bull / Bear Cards — white base, 3px left signal only ──────── */
.bull-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--signal-bull);
    border-radius: 6px;
    padding: 11px 15px;
    margin-bottom: 8px;
    font-family: var(--mono);
    font-size: 0.84rem;
    color: var(--text-primary);
    line-height: 1.65;
}
.bear-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--signal-bear);
    border-radius: 6px;
    padding: 11px 15px;
    margin-bottom: 8px;
    font-family: var(--mono);
    font-size: 0.84rem;
    color: var(--text-primary);
    line-height: 1.65;
}

/* ── Signal Badge Pills ─────────────────────────────────────────── */
.signal-badge-bull,
.signal-badge-bear {
    display: inline-block;
    background: transparent;
    border-radius: 3px;
    padding: 2px 10px;
    font-family: var(--mono);
    font-size: 0.63rem;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    margin-bottom: 12px;
    border: 1px solid;
}
.signal-badge-bull { color: var(--signal-bull); border-color: var(--signal-bull); }
.signal-badge-bear { color: var(--signal-bear); border-color: var(--signal-bear); }

/* ── Consensus Bar ──────────────────────────────────────────────── */
.consensus-bar {
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-left: 3px solid var(--border-strong);
    border-radius: 6px;
    padding: 9px 13px;
    font-family: var(--mono);
    font-size: 0.80rem;
    color: var(--text-secondary);
    margin-bottom: 14px;
    line-height: 1.55;
}

/* ── Two-Stage DCF Badge ────────────────────────────────────────── */
.two-stage-badge {
    display: block;
    background: var(--bg-subtle);
    border-left: 2px solid var(--accent);
    border-radius: 4px;
    padding: 6px 11px;
    font-family: var(--mono);
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.08em;
    margin-bottom: 10px;
}

/* ── Report Body ────────────────────────────────────────────────── */
.report-body {
    font-family: var(--mono);
    font-size: 0.87rem;
    line-height: 1.72;
    color: var(--text-primary);
}
.report-body h3, .report-body h4 {
    font-family: var(--mono) !important;
    color: var(--text-primary) !important;
    margin: 20px 0 6px !important;
}
.report-body h3 { font-size: 0.88rem !important; font-weight: 700 !important; }
.report-body h4 { font-size: 0.84rem !important; font-weight: 600 !important; color: var(--text-secondary) !important; }
.report-body blockquote {
    border-left: 2px solid var(--border-strong);
    padding-left: 13px;
    margin: 10px 0;
    color: var(--text-secondary);
    font-style: normal;
}
.report-body code {
    font-family: var(--mono);
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 1px 5px;
    font-size: 0.85em;
    color: var(--text-primary);
}
.report-body hr { border-top: 1px solid var(--border) !important; margin: 16px 0 !important; }

/* ── Tabs ───────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: var(--mono) !important;
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    padding: 8px 16px !important;
    letter-spacing: 0.04em !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom-color: var(--accent) !important;
    font-weight: 700 !important;
}

/* ── st.metric ──────────────────────────────────────────────────── */
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    white-space: nowrap !important;
    color: var(--text-primary) !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--mono) !important;
    font-size: 0.63rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    white-space: nowrap !important;
    color: var(--text-muted) !important;
}
[data-testid="stMetricDelta"] svg { display: none !important; }
[data-testid="stMetricDelta"] { font-family: var(--mono) !important; font-size: 0.70rem !important; }

/* ── DataFrame ──────────────────────────────────────────────────── */
[data-testid="stDataFrame"] * { font-family: var(--mono) !important; font-size: 0.80rem !important; }

/* ── Progress Bar — owns its vertical space, no HTML above/below ── */
[data-testid="stProgressBar"] {
    display: block !important;
    margin: 10px 0 16px !important;
    clear: both !important;
}
[data-testid="stProgressBar"] > div {
    background: var(--bg-overlay) !important;
    border-radius: 4px !important;
    height: 6px !important;
    overflow: hidden !important;
}
[data-testid="stProgressBar"] > div > div {
    background: var(--accent) !important;
    border-radius: 4px !important;
}

/* ── Alert Boxes ────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    background: var(--bg-card) !important;
    border-radius: 6px !important;
    font-family: var(--mono) !important;
    font-size: 0.84rem !important;
}

/* ── Buttons ────────────────────────────────────────────────────── */
[data-testid="stButton"] button {
    font-family: var(--mono) !important;
    font-size: 0.79rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    border-radius: 5px !important;
    border: 1px solid var(--border-strong) !important;
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    transition: border-color 0.12s ease, background 0.12s ease;
}
[data-testid="stButton"] button:hover {
    border-color: var(--accent) !important;
    background: var(--bg-subtle) !important;
}

/* ── Expanders ──────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
[data-testid="stExpander"] summary {
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    color: var(--text-secondary) !important;
}
/* Keep expander toggle icon at fixed width so it doesn't bleed into title */
[data-testid="stExpander"] summary svg,
[data-testid="stExpanderToggleIcon"] {
    flex-shrink: 0 !important;
    width: 1rem !important;
    height: 1rem !important;
}

/* ── Sidebar widget labels ──────────────────────────────────────── */
[data-testid="stSelectbox"] label,
[data-testid="stRadio"] label,
[data-testid="stTextInput"] label,
[data-testid="stSlider"] label {
    font-family: var(--mono) !important;
    font-size: 0.70rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
}

/* ── Chat bubbles ───────────────────────────────────────────────── */
[data-testid="stChatMessage"] { padding: 4px 0 !important; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
    line-height: 1.65 !important;
    color: var(--text-primary) !important;
}

/* ── Brand header block ─────────────────────────────────────────── */
.brand-header {
    font-family: var(--mono);
    margin: 0 0 6px;
    padding: 0;
    line-height: 1;
}
.brand-header h1 {
    margin: 0 0 5px !important;
    padding: 0 !important;
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
    white-space: nowrap !important;
    /* Override Streamlit h1 size reset */
    font-family: var(--mono) !important;
}
.brand-header .brand-sub {
    font-size: 0.95rem;
    font-weight: 400;
    color: var(--text-secondary);
    margin-left: 12px;
    letter-spacing: 0;
}
.brand-header .brand-meta {
    font-size: 0.74rem;
    color: var(--text-muted);
    letter-spacing: 0.04em;
    margin-top: 3px;
}
.brand-header .brand-ticker {
    font-weight: 700;
    color: var(--accent);
    font-size: 0.85rem;
    margin-right: 10px;
}
.brand-header .brand-tag {
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1px 8px;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-right: 8px;
}

/* ── MoS label row ──────────────────────────────────────────────── */
.mos-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin: 10px 0 4px;
    font-family: var(--mono);
}
.mos-row .mos-lbl {
    font-size: 0.74rem;
    color: var(--text-secondary);
}
.mos-row .mos-val {
    font-size: 1.05rem;
    font-weight: 700;
    white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# 国际化文本字典 / i18n Translation Dict
# ══════════════════════════════════════════════════════════════════
_I18N = {
    "简体中文": {
        # Sidebar
        "lang_label":        "Language / 语言",
        "ticker_label":      "股票代码",
        "ticker_help":       "输入美股代码，支持 yfinance 覆盖的所有标的。",
        "mode_label":        "分析模式",
        "mode_live":         "📡  实时深度分析",
        "mode_bt":           "🕰️  历史时空盲测",
        "bt_year_label":     "回测节点年份",
        "run_btn":           "🚀  开始深度透视",
        "disclaimer":        "⚠️ 本看板仅供研究参考，不构成投资建议。",
        # Main header
        "live_mode_tag":     "📡 实时深度分析模式",
        "today_label":       "今日",
        "bt_mode_tag":       "🕰️ 历史时空盲测模式",
        "snapshot_label":    "回测节点",
        # Prompt
        "prompt_info":       "👈 在左侧输入股票代码并点击「开始深度透视」按钮。",
        "no_ticker_warn":    "请先输入股票代码。",
        # Company card
        "company_card_title":"公司基本面名片",
        "sector":            "板块",
        "industry":          "行业",
        "exchange":          "交易所",
        "ipo_date":          "上市日期",
        "employees":         "员工数",
        "website":           "官网",
        "current_price":     "当前股价",
        "market_cap":        "总市值",
        "week52":            "52周区间",
        "summary_expand":    "业务简介（点击展开）",
        "no_info":           "⚠️ 无法获取公司信息。",
        # Dashboard
        "quant_dash":        "📐 核心量化仪表盘",
        "roe_label":         "ROE · 净资产回报率",
        "roic_label":        "ROIC · 投入资本回报",
        "gm_label":          "毛利率 · Gross Margin",
        "de_label":          "D/E · 资产负债率",
        "fcf_label":         "FCF · 自由现金流",
        "roe_strong":        "优秀",  "roe_mid": "一般",  "roe_weak": "偏弱",
        "roic_strong":       "优秀",  "roic_mid":"一般",  "roic_weak":"偏弱",
        "gm_strong":         "护城河","gm_mid":  "一般",  "gm_weak": "偏低",
        "de_low":            "低杠杆","de_mid":  "中等",  "de_high": "高杠杆",
        "fcf_pos":           "正",    "fcf_neg": "负",
        "trend_expand":      "📅 近 5 年指标趋势",
        # MoS
        "mos_canvas":        "🎯 安全边际画布",
        "market_price":      "当前股价 · Market Price",
        "intrinsic_label":   "AI 内在价值 · Intrinsic Value",
        "mos_progress_lbl":  "安全边际 Margin of Safety",
        "dcf_params":        "DCF 模型参数",
        "wacc_label":        "折现率 · WACC",
        "term_g_label":      "永续增长 · Terminal g",
        "fcf_cagr_label":    "FCF 增速 · FCF CAGR",
        "stage1_label":      "Agent 预测 · 前5年增速",
        "stage2_label":      "Agent 预测 · 第6年起始",
        "two_stage_badge":   "🤖 两阶段 DCF 已激活",
        "dcf_err_label":     "DCF 计算异常",
        "mos_deep":          "✅ **显著低估 · Deep Value**\n\n充足安全边际，符合价值投资买入标准。",
        "mos_slight":        "🟡 **轻微低估 · Slight Discount**\n\n有一定安全边际，可分批布局。",
        "mos_fair":          "⚪ **基本合理 · Fair Value**\n\n估值接近内在价值，建议继续观察。",
        "mos_premium":       "🟠 **轻微高估 · Slight Premium**\n\n当前价格高于内在价值，需谨慎。",
        "mos_over":          "🔴 **显著高估 · Overvalued**\n\n价格大幅超出内在价值，风险较高。",
        # Agent panel
        "agent_panel":       "🤖 Agent 思维链与报告",
        "bull_tab":          "📈 多方论点",
        "bear_tab":          "📉 空方论点",
        "report_tab":        "📄 投委会深度报告",
        "consensus_label":   "市场共识",
        "no_bull":           "暂无多方数据。",
        "no_bear":           "暂无空方数据。",
        "report_empty":      "报告内容为空。",
        "report_err":        "报告生成失败",
        # Backtest
        "bt_snap":           "📐 历史量化快照",
        "bt_mos":            "🎯 历史安全边际",
        "entry_price":       "年底入场价",
        "hist_iv":           "历史 DCF 内在价值",
        "return_cmp":        "📊 持有收益对比",
        "held_note":         "年底买入 → 持有至今",
        "asset":             "资产",
        "entry_col":         "入场价",
        "today_col":         "今日价",
        "return_col":        "总回报",
        "buy_correct":       "✅ **BUY 正确 · Correct Call**",
        "buy_wrong":         "🔴 **BUY 失误 · Incorrect Call**",
        "sell_correct":      "✅ **SELL 正确 · Correct Call**",
        "sell_miss":         "🟠 **SELL 失策 · Missed Opportunity**",
        "hold_neutral":      "🟡 **HOLD 观望 · Neutral Call**",
        "bt_agent_panel":    "🤖 历史 Agent 分析链",
        "macro_tab":         "🌐 宏观背景",
        "hist_report_tab":   "📄 历史投委会报告",
        "macro_label":       "宏观/行业背景",
        "download_btn":      "⬇️  下载完整盲测报告 (.md)",
        "bt_insuf":          "历史财报切片数据不足。",
        "bt_empty":          "盲测返回空结果。",
        "ai_decision":       "AI 投资决策（回测节点年底）",
        # Fetching
        "fetch_spin":        "⏳ 正在拉取财务数据…",
        "fetch_err":         "数据拉取失败",
        "broker_spin":       "获取市场情报…",
        "report_spin":       "⚙️ 投委会合成报告中…",
        "bt_spin":           "⏳ 正在加载历史盲测…",
        "bt_err":            "盲测失败",
        # Footer
        "footer":            "🏛️ **问巴菲特与达利欧** · 由 yfinance + OpenAI GPT-4o-mini 驱动 · ⚠️ 仅供研究参考，不构成投资建议",
        # Outperform
        "outperform":        "跑赢 vs SPY",
        "inline":            "持平 vs SPY",
        "underperform":      "跑输 vs SPY",
        # Interactive Chat
        "chat_title":        "💬 与投委会实时对话（达利欧风格）",
        "chat_subtitle":     "基于当前量化数据与五步法报告，向 AI 投资顾问提问",
        "chat_placeholder":  "例如：这个估值区间合理吗？最大的下行风险是什么？",
        "chat_thinking":     "🤔 投委会分析中…",
        "chat_no_data":      "请先在上方点击【开始分析】生成报告，再开启对话。",
        "chat_welcome":      "你好！我是【问巴菲特与达利欧】的 AI 投资顾问，融合巴菲特价值投资哲学与达利欧系统机器思维。\n\n我已读取 **{ticker}** 的所有量化指标与五步法评估报告。请直接提问——估值合理性、风险识别、宏观周期定位，或任何你关心的投资问题。",
    },
    "English": {
        "lang_label":        "Language / 语言",
        "ticker_label":      "Ticker Symbol",
        "ticker_help":       "Enter any US ticker supported by yfinance.",
        "mode_label":        "Analysis Mode",
        "mode_live":         "📡  Live Deep Analysis",
        "mode_bt":           "🕰️  Historical Backtest",
        "bt_year_label":     "Backtest Snapshot Year",
        "run_btn":           "🚀  Analyze",
        "disclaimer":        "⚠️ For research purposes only. Not financial advice.",
        "live_mode_tag":     "📡 Live Analysis Mode",
        "today_label":       "Today",
        "bt_mode_tag":       "🕰️ Backtest Mode",
        "snapshot_label":    "Snapshot Year",
        "prompt_info":       "👈 Enter a ticker in the sidebar and click Analyze to begin.",
        "no_ticker_warn":    "Please enter a ticker symbol.",
        "company_card_title":"Company Profile",
        "sector":            "Sector",
        "industry":          "Industry",
        "exchange":          "Exchange",
        "ipo_date":          "IPO Date",
        "employees":         "Employees",
        "website":           "Website",
        "current_price":     "Current Price",
        "market_cap":        "Market Cap",
        "week52":            "52-Week Range",
        "summary_expand":    "Business Summary (click to expand)",
        "no_info":           "⚠️ Unable to retrieve company info.",
        "quant_dash":        "📐 Quantitative Dashboard",
        "roe_label":         "ROE · Return on Equity",
        "roic_label":        "ROIC · Return on Inv. Capital",
        "gm_label":          "Gross Margin",
        "de_label":          "D/E · Debt-to-Equity",
        "fcf_label":         "FCF · Free Cash Flow",
        "roe_strong":        "Strong", "roe_mid": "Moderate", "roe_weak": "Weak",
        "roic_strong":       "Strong", "roic_mid":"Moderate", "roic_weak":"Weak",
        "gm_strong":         "Moat",   "gm_mid":  "Moderate", "gm_weak": "Low",
        "de_low":            "Low Lev","de_mid":  "Mid",       "de_high": "High Lev",
        "fcf_pos":           "Positive","fcf_neg": "Negative",
        "trend_expand":      "📅 5-Year Metrics Trend",
        "mos_canvas":        "🎯 Margin of Safety",
        "market_price":      "Market Price",
        "intrinsic_label":   "AI Intrinsic Value",
        "mos_progress_lbl":  "Margin of Safety",
        "dcf_params":        "DCF Model Parameters",
        "wacc_label":        "Discount Rate (WACC)",
        "term_g_label":      "Terminal Growth",
        "fcf_cagr_label":    "FCF CAGR",
        "stage1_label":      "Agent Yr 1-5 Growth",
        "stage2_label":      "Agent Yr 6 Transition",
        "two_stage_badge":   "🤖 Two-Stage DCF Active",
        "dcf_err_label":     "DCF Calculation Error",
        "mos_deep":          "✅ **Deep Value**\n\nStrong margin of safety — meets value-buying criteria.",
        "mos_slight":        "🟡 **Slight Discount**\n\nModerate margin — consider gradual entry.",
        "mos_fair":          "⚪ **Fair Value**\n\nNear fair value — monitor and wait.",
        "mos_premium":       "🟠 **Slight Premium**\n\nSlightly overvalued — proceed with caution.",
        "mos_over":          "🔴 **Overvalued**\n\nSignificantly overvalued — high risk.",
        "agent_panel":       "🤖 Agent Reasoning & Report",
        "bull_tab":          "📈 Bull Case",
        "bear_tab":          "📉 Bear Case",
        "report_tab":        "📄 Full Committee Report",
        "consensus_label":   "Market Consensus",
        "no_bull":           "No bullish data available.",
        "no_bear":           "No bearish data available.",
        "report_empty":      "Report is empty.",
        "report_err":        "Report generation failed",
        "bt_snap":           "📐 Historical Snapshot",
        "bt_mos":            "🎯 Historical MoS",
        "entry_price":       "Entry Price",
        "hist_iv":           "Historical DCF Intrinsic Value",
        "return_cmp":        "📊 Return Comparison",
        "held_note":         "Held from year-end to today",
        "asset":             "Asset",
        "entry_col":         "Entry Price",
        "today_col":         "Today",
        "return_col":        "Total Return",
        "buy_correct":       "✅ **Correct BUY Call**",
        "buy_wrong":         "🔴 **Incorrect BUY Call**",
        "sell_correct":      "✅ **Correct SELL Call**",
        "sell_miss":         "🟠 **Missed Upside (SELL)**",
        "hold_neutral":      "🟡 **Neutral HOLD Call**",
        "bt_agent_panel":    "🤖 Historical Agent Reasoning",
        "macro_tab":         "🌐 Macro Context",
        "hist_report_tab":   "📄 Historical Committee Report",
        "macro_label":       "Macro & Industry Context",
        "download_btn":      "⬇️  Download Full Backtest Report (.md)",
        "bt_insuf":          "Insufficient historical data for this snapshot.",
        "bt_empty":          "Backtest returned empty result.",
        "ai_decision":       "AI Investment Decision (at year-end snapshot)",
        "fetch_spin":        "⏳ Fetching financial data…",
        "fetch_err":         "Data fetch failed",
        "broker_spin":       "Fetching market intelligence…",
        "report_spin":       "⚙️ Committee synthesizing report…",
        "bt_spin":           "⏳ Running historical backtest…",
        "bt_err":            "Backtest failed",
        "footer":            "🏛️ **Ask Buffett & Dalio** · Powered by yfinance & OpenAI GPT-4o-mini · ⚠️ For research only, not financial advice",
        "outperform":        "Outperformed SPY",
        "inline":            "In-line with SPY",
        "underperform":      "Underperformed SPY",
        # Interactive Chat
        "chat_title":        "💬 Chat with the Investment Committee (Dalio Style)",
        "chat_subtitle":     "Ask the AI advisor anything based on the current data & Five-Step report",
        "chat_placeholder":  "e.g. Is this valuation reasonable? What's the biggest downside risk?",
        "chat_thinking":     "🤔 Committee analysing…",
        "chat_no_data":      "Please run an analysis above first, then start chatting.",
        "chat_welcome":      "Hello! I'm the AI advisor for **Ask Buffett & Dalio** — combining Buffett's value investing philosophy with Dalio's Systems/Machine Thinking.\n\nI've reviewed all the quantitative metrics and the Five-Step assessment for **{ticker}**. Ask me anything — valuation range, risk identification, macro cycle positioning, or any investment question you have.",
    },
}


# ══════════════════════════════════════════════════════════════════
# 侧边栏（语言切换必须最先渲染）
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    # Custom brand replaces the native sidebar header (which shows garbled Material Icon text)
    st.markdown(
        '<div class="sidebar-brand">'
        '<span class="sidebar-brand-icon">📊</span>'
        '<span class="sidebar-brand-text">Ask Buffett &amp; Dalio<br>'
        '<span style="font-weight:400;opacity:0.6;">问巴菲特与达利欧</span></span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Language toggle — always first so T is available everywhere below
    lang = st.radio(
        "Language / 语言",
        options=["简体中文", "English"],
        horizontal=True,
        key="lang",
    )
    T         = _I18N[lang]
    lang_code = "en" if lang == "English" else "zh"

    st.divider()

    ticker_input = st.text_input(
        T["ticker_label"],
        value="AAPL",
        placeholder="e.g. AAPL, MSFT, NVDA",
        help=T["ticker_help"],
    ).strip().upper()

    mode = st.radio(
        T["mode_label"],
        options=[T["mode_live"], T["mode_bt"]],
        index=0,
    )
    is_backtest = mode.startswith("🕰️")

    backtest_year = None
    if is_backtest:
        backtest_year = st.slider(
            T["bt_year_label"],
            min_value=2018,
            max_value=date.today().year - 1,
            value=2023,
        )

    st.divider()
    run_btn = st.button(T["run_btn"], use_container_width=True, type="primary")
    st.divider()
    st.caption(T["disclaimer"])


# ══════════════════════════════════════════════════════════════════
# 后端模块导入
# 不用 @st.cache_resource —— 它会锁住旧版模块对象，导致代码改动后
# 新参数（如 lang=）对缓存的旧模块不可见，调用时抛 TypeError。
# Python 的 sys.modules 本身已做模块级缓存，无需额外包装。
# ══════════════════════════════════════════════════════════════════
def _import_backends():
    try:
        import financial_engine as fe
        import analysis_agents  as aa
        import backtester       as bt
        return fe, aa, bt, None
    except Exception as e:
        return None, None, None, str(e)


# ══════════════════════════════════════════════════════════════════
# 缓存数据拉取
# ══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=3600)
def load_company_info(ticker: str) -> dict:
    """
    拉取公司信息，供名片模块使用。
    三层降级策略：
      1. fast_info  — 价格/市值/52周区间（最可靠）
      2. history_metadata — 公司名称/交易所/上市日期（调用 history 时自动填充）
      3. tk.info    — 完整信息（sector/employees/website 等），可能因 curl_cffi bug 失败
    """
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        merged = {}

        def _sa(obj, attr):
            try:
                v = getattr(obj, attr)
                return v if v is not None else None
            except Exception:
                return None

        # ── 层 1: fast_info（价格类，最稳定）──────────────
        fi = getattr(tk, "fast_info", None)
        if fi:
            merged.update({
                "currentPrice":     _sa(fi, "last_price"),
                "marketCap":        _sa(fi, "market_cap"),
                "currency":         _sa(fi, "currency"),
                "fiftyTwoWeekHigh": _sa(fi, "year_high"),
                "fiftyTwoWeekLow":  _sa(fi, "year_low"),
                "exchange":         _sa(fi, "exchange"),
                "sharesOutstanding":_sa(fi, "shares"),
            })

        # ── 层 2: history_metadata（触发 history 调用填充元数据）──
        try:
            tk.history(period="5d")          # 触发 metadata 填充
            meta = getattr(tk, "history_metadata", {}) or {}
            if meta.get("shortName") and not merged.get("shortName"):
                merged["shortName"] = meta["shortName"]
            if meta.get("longName") and not merged.get("longName"):
                merged["longName"] = meta["longName"]
            if meta.get("exchangeName") and not merged.get("fullExchangeName"):
                merged["fullExchangeName"] = meta["exchangeName"]
            if meta.get("currency") and not merged.get("currency"):
                merged["currency"] = meta["currency"]
            # history_metadata 中的 firstTradeDate（unix 秒）
            ftd = meta.get("firstTradeDate") or meta.get("firstTradeDateEpochUtc")
            if ftd and not merged.get("firstTradeDateEpochUtc"):
                merged["firstTradeDateEpochUtc"] = ftd
        except Exception:
            pass

        # ── 层 3: tk.info（完整数据，可能失败）─────────────
        try:
            full = tk.info or {}
            if isinstance(full, dict):
                for k in ("longName", "shortName", "longBusinessSummary",
                          "sector", "industry", "country", "fullExchangeName",
                          "fullTimeEmployees", "website",
                          "firstTradeDateEpochUtc", "ipoDate"):
                    if full.get(k) is not None and not merged.get(k):
                        merged[k] = full[k]
                # 价格类 fallback（fast_info 失败时补充）
                for k in ("currentPrice", "regularMarketPrice",
                          "marketCap", "fiftyTwoWeekHigh", "fiftyTwoWeekLow"):
                    if not merged.get(k) and full.get(k):
                        merged[k] = full[k]
        except Exception:
            pass   # curl_cffi bug 时静默跳过，用已有数据

        return merged
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=3600)
def load_realtime(ticker: str):
    fe, _, _, err = _import_backends()
    if err:
        return None, None, err
    try:
        data    = fe.fetch_financials(ticker)
        metrics = fe.calculate_metrics(data)
        dcf     = fe.calculate_dcf_value(ticker, _data=data)
        return metrics, dcf, None
    except Exception as e:
        return None, None, str(e)


@st.cache_data(show_spinner=False, ttl=3600)
def load_broker_views(ticker: str, lang_code: str = "zh"):
    _, aa, _, err = _import_backends()
    if err:
        return None, err
    try:
        views = aa.fetch_broker_views(ticker, lang=lang_code)
        # 如果 LLM 返回空或只有错误占位符，做一次降级
        if not views or not views.get("bullish"):
            views = aa.fetch_broker_views(ticker, lang="zh")  # 降级到中文本地库
        return views, None
    except TypeError:
        # analysis_agents 旧版缓存无 lang 参数时的兜底
        try:
            return aa.fetch_broker_views(ticker), None
        except Exception as e2:
            return None, str(e2)
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner=False, ttl=3600)
def load_full_analysis(ticker: str, metrics_json: str, dcf_json: str,
                       lang_code: str = "zh"):
    _, aa, _, err = _import_backends()
    if err:
        return None, None, None, None, err
    try:
        result = aa.run_full_analysis(
            ticker,
            json.loads(metrics_json),
            json.loads(dcf_json),
            use_search=False,
            save_report=False,
            lang=lang_code,
        )
        # run_full_analysis returns (report, stage1, stage2, updated_dcf)
        if isinstance(result, tuple) and len(result) == 4:
            report, stage1, stage2, updated_dcf = result
        else:
            # backward-compat: old version returned just a string
            report, stage1, stage2, updated_dcf = result, None, None, {}
        return report, stage1, stage2, updated_dcf, None
    except Exception as e:
        return None, None, None, None, str(e)


@st.cache_data(show_spinner=False, ttl=3600)
def load_backtest(ticker: str, backtest_year: int):
    _, _, bt, err = _import_backends()
    if err:
        return None, err
    try:
        return bt.run_backtest(ticker, backtest_year=backtest_year, save_report=False), None
    except Exception as e:
        return None, str(e)


# NOTE: chat replies must NOT be cached globally — each turn is unique.
# We call analysis_agents.answer_investor_question() directly in the UI loop.
def _call_chat_answer(user_query: str, ticker: str,
                      metrics_json: str, dcf_json: str,
                      agent_report: str, history: list,
                      lang_code: str) -> str:
    """薄封装：从 backend 调用 answer_investor_question，隔离 import 错误。"""
    _, aa, _, err = _import_backends()
    if err:
        return f"[Backend 加载失败] {err}"
    try:
        return aa.answer_investor_question(
            user_query   = user_query,
            ticker       = ticker,
            metrics_json = json.loads(metrics_json) if isinstance(metrics_json, str) else metrics_json,
            dcf_json     = json.loads(dcf_json)     if isinstance(dcf_json, str)     else dcf_json,
            agent_report = agent_report,
            chat_history = history,
            lang         = lang_code,
        )
    except Exception as e:
        return f"[对话异常] {e}"


# ══════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════
def _fmt_price(v, currency="$"):
    return f"{currency}{v:,.2f}" if v is not None else "N/A"

def _fmt_pct(v, decimals=1):
    if v is None:
        return "N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{decimals}f}%"

def _safe(d: dict, *keys, default="N/A"):
    val = d
    for k in keys:
        try:
            val = val[k]
            if val is None:
                return default
        except (KeyError, TypeError):
            return default
    if isinstance(val, float) and math.isnan(val):
        return default
    return val

def _latest_metric(df: pd.DataFrame, col: str):
    if df is None or df.empty or col not in df.columns:
        return None
    s = df[col].dropna()
    return float(s.iloc[-1]) if not s.empty else None

def _decision_badge(decision: str) -> str:
    icons  = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}
    labels = {"BUY": "BUY · 买入", "HOLD": "HOLD · 持有", "SELL": "SELL · 卖出"}
    cls    = {"BUY": "badge-buy",  "HOLD": "badge-hold",   "SELL": "badge-sell"}
    return (f'<span class="{cls.get(decision,"badge-hold")}">'
            f'{icons.get(decision,"⚪")} &nbsp; {labels.get(decision,decision)}</span>')

def _delta_color(v, threshold=0):
    if v is None: return "off"
    return "normal" if v >= threshold else "inverse"

def _kpi(label: str, value: str, col=None):
    """
    KPI card: label (small-caps) + value (mono bold, nowrap).
    Renders into `col` if given, otherwise into current st context.
    No inline sizing — all layout driven by .kpi-box CSS class.
    """
    html = (
        '<div class="kpi-box">'
        f'<span class="kpi-label">{label}</span>'
        f'<span class="kpi-value">{value}</span>'
        '</div>'
    )
    (col if col is not None else st).markdown(html, unsafe_allow_html=True)


def _price_row(market_price: str, intrinsic_value: str,
               price_label: str, iv_label: str):
    """
    MoS canvas price pair — two equal columns via st.columns.
    No HTML flexbox; avoids stripe-overlap with st.progress.
    """
    ca, cb = st.columns(2, gap="small")
    ca.markdown(
        '<div class="price-card">'
        f'<span class="price-card-label">{price_label}</span>'
        f'<span class="price-card-value">{market_price}</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    cb.markdown(
        '<div class="price-card">'
        f'<span class="price-card-label">{iv_label}</span>'
        f'<span class="price-card-value intrinsic">{intrinsic_value}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

def _downgrade_headers(md: str) -> str:
    """将报告中 # → ###，## → #### 减小 Streamlit 渲染字号。"""
    md = re.sub(r'^#{1}\s', '### ', md, flags=re.MULTILINE)
    md = re.sub(r'^#{2}\s', '#### ', md, flags=re.MULTILINE)
    return md


# ══════════════════════════════════════════════════════════════════
# 公司基本面名片组件
# ══════════════════════════════════════════════════════════════════
def render_company_card(info: dict, ticker: str, T: dict):
    if not info:
        st.warning(T["no_info"])
        return

    name     = info.get("longName") or info.get("shortName") or ticker
    sector   = info.get("sector",   "—")
    industry = info.get("industry", "—")
    exchange = info.get("fullExchangeName") or info.get("exchange", "—")
    currency = info.get("currency", "USD")
    employees= info.get("fullTimeEmployees")
    website  = info.get("website", "")
    summary  = info.get("longBusinessSummary", "")

    # IPO 日期
    ipo_str = "—"
    if info.get("ipoDate"):
        ipo_str = str(info["ipoDate"])
    elif info.get("firstTradeDateEpochUtc"):
        try:
            ipo_str = datetime.fromtimestamp(
                info["firstTradeDateEpochUtc"], tz=timezone.utc
            ).strftime("%Y-%m-%d")
        except Exception:
            pass

    price   = info.get("currentPrice") or info.get("regularMarketPrice")
    mktcap  = info.get("marketCap")
    wk52h   = info.get("fiftyTwoWeekHigh")
    wk52l   = info.get("fiftyTwoWeekLow")

    price_str = f"{currency} {price:,.2f}" if price else "N/A"
    mcap_str  = f"${mktcap/1e9:,.1f} B"   if mktcap else "N/A"
    wk52_str  = (f"{wk52l:,.2f} – {wk52h:,.2f}" if wk52h and wk52l else "N/A")
    emp_str   = f"{employees:,}" if employees else "—"

    # ── 名片 HTML ─────────────────────────────────────────────────
    tags_html = "".join(
        f'<span class="company-tag">{t}</span>'
        for t in [sector, industry, exchange]
        if t and t != "—"
    )
    _website_html = (
        f"<span>🌐 <b>{T['website']}</b>: "
        f"<a href='{website}' target='_blank' "
        f"style='color:var(--accent);text-decoration:none;'>"
        f"{website.replace('https://','').rstrip('/')}</a></span>"
        if website else ""
    )
    st.markdown(f"""
<div class="company-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
    <div style="min-width:0;">
      <div class="company-name">{name}</div>
      <div class="company-meta">{ticker} &nbsp;·&nbsp; {exchange}</div>
      <div style="margin-top:8px;">{tags_html}</div>
    </div>
    <div style="text-align:right;flex-shrink:0;">
      <div class="price-big">{price_str}</div>
      <div style="font-family:var(--mono);font-size:0.72rem;color:var(--text-muted);margin-top:3px;">
        {T['market_cap']}: {mcap_str}
      </div>
      <div style="font-family:var(--mono);font-size:0.72rem;color:var(--text-muted);white-space:nowrap;">
        52W: {wk52_str}
      </div>
    </div>
  </div>
  <hr style="border-color:var(--border);margin:14px 0 10px;">
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:6px;
              font-family:var(--mono);font-size:0.75rem;color:var(--text-secondary);">
    <span>📅 {T['ipo_date']}: <b style="color:var(--text-primary);">{ipo_str}</b></span>
    <span>👥 {T['employees']}: <b style="color:var(--text-primary);">{emp_str}</b></span>
    {_website_html}
  </div>
</div>
""", unsafe_allow_html=True)

    # ── 业务简介（折叠）────────────────────────────────────────────
    if summary:
        with st.expander(T["summary_expand"], expanded=False):
            st.markdown(
                f"<div style='font-family:var(--mono);font-size:0.84rem;"
                f"color:var(--text-secondary);line-height:1.7;'>{summary}</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════
# 主面板标题
# ══════════════════════════════════════════════════════════════════
_mode_tag  = T["bt_mode_tag"]  if is_backtest else T["live_mode_tag"]
_mode_meta = (f"{T['snapshot_label']}: {backtest_year}"
              if is_backtest else f"{T['today_label']}: {date.today().isoformat()}")

# ── 全局大标题（主面板最顶部，宽度充足，不折行）─────────────────────
st.markdown(
    f'<div class="brand-header">'
    f'<h1>📊 问巴菲特与达利欧'
    f'<span class="brand-sub">Ask Buffett &amp; Dalio</span>'
    f'</h1>'
    f'<div class="brand-meta">'
    f'<span class="brand-ticker">{ticker_input or "—"}</span>'
    f'<span class="brand-tag">{_mode_tag}</span>'
    f'<span>{_mode_meta}</span>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)
st.divider()

# ── 等待按钮 ─────────────────────────────────────────────────────
# Allow chat reruns to bypass the gate when analysis was already completed for this ticker
_analysis_done = st.session_state.get("_analysis_done_ticker") == ticker_input
if not run_btn and not _analysis_done:
    st.info(T["prompt_info"], icon="💡")
    st.stop()

if not ticker_input:
    st.warning(T["no_ticker_warn"], icon="⚠️")
    st.stop()

# 每次用户触发新分析时，清除旧 Agent 预测，防止跨 ticker 残留
_prev_ticker = st.session_state.get("_last_ticker", "")
if ticker_input != _prev_ticker:
    for _k in ("_stage1_growth", "_stage2_growth_start", "_updated_dcf", "_analysis_done_ticker"):
        st.session_state.pop(_k, None)
    st.session_state["_last_ticker"] = ticker_input

# Persist the "analysis was run" flag so chat reruns skip the gate above
if run_btn:
    st.session_state["_analysis_done_ticker"] = ticker_input
_analysis_done = st.session_state.get("_analysis_done_ticker") == ticker_input


# ══════════════════════════════════════════════════════════════════
# ── 实时分析模式 ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════
if not is_backtest:

    # ── 公司名片 ──────────────────────────────────────────────────
    with st.spinner(f"🔍 {ticker_input}…"):
        company_info = load_company_info(ticker_input)

    st.markdown(f"### 🏢 {T['company_card_title']}")
    render_company_card(company_info, ticker_input, T)
    st.divider()

    # ── 财务数据拉取 ──────────────────────────────────────────────
    with st.spinner(T["fetch_spin"]):
        metrics_df, dcf, err_data = load_realtime(ticker_input)

    if err_data:
        st.error(f"{T['fetch_err']}: `{err_data}`")
        st.stop()

    # ────────────────────────────────────────────────────────────
    # 模块一：核心量化仪表盘
    # ────────────────────────────────────────────────────────────
    st.markdown(f"### {T['quant_dash']}")

    roe  = _latest_metric(metrics_df, "ROE (%)")
    roic = _latest_metric(metrics_df, "ROIC (%)")
    gm   = _latest_metric(metrics_df, "Gross Margin (%)")
    de   = _latest_metric(metrics_df, "D/E Ratio")
    fcf  = _latest_metric(metrics_df, "FCF ($M)")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(T["roe_label"],
              f"{roe:.1f}%" if roe is not None else "N/A",
              delta=T["roe_strong"] if roe and roe>=15 else (T["roe_mid"] if roe and roe>=8 else T["roe_weak"]),
              delta_color=_delta_color(roe,15),
              help="Return on Equity. ≥15% = Strong.")
    c2.metric(T["roic_label"],
              f"{roic:.1f}%" if roic is not None else "N/A",
              delta=T["roic_strong"] if roic and roic>=12 else (T["roic_mid"] if roic and roic>=6 else T["roic_weak"]),
              delta_color=_delta_color(roic,12),
              help="Return on Invested Capital. ≥12% = Strong.")
    c3.metric(T["gm_label"],
              f"{gm:.1f}%" if gm is not None else "N/A",
              delta=T["gm_strong"] if gm and gm>=40 else (T["gm_mid"] if gm and gm>=20 else T["gm_weak"]),
              delta_color=_delta_color(gm,40),
              help="Gross Profit Margin. ≥40% = Pricing power.")
    c4.metric(T["de_label"],
              f"{de:.2f}x" if de is not None else "N/A",
              delta=T["de_low"] if de and de<1 else (T["de_mid"] if de and de<2 else T["de_high"]),
              delta_color="normal" if de and de<1 else ("off" if de and de<2 else "inverse"),
              help="Total Debt / Equity. Lower = more conservative.")

    # FCF 用自定义卡片防止大数折行
    with c5:
        fcf_str = (f"${fcf/1000:,.1f}B" if fcf and abs(fcf)>=1000
                   else (f"${fcf:,.0f}M" if fcf is not None else "N/A"))
        _kpi(T["fcf_label"], fcf_str)

    if metrics_df is not None and not metrics_df.empty:
        with st.expander(T["trend_expand"], expanded=False):
            fmt = {
                "ROE (%)": "{:.1f}%", "ROIC (%)": "{:.1f}%",
                "Gross Margin (%)": "{:.1f}%",
                "FCF ($M)": "{:,.0f}", "D/E Ratio": "{:.2f}x",
            }
            try:
                # background_gradient requires matplotlib — graceful fallback
                styled = metrics_df.style.format(fmt).background_gradient(
                    cmap="RdYlGn", axis=None
                )
                st.dataframe(styled, use_container_width=True)
            except Exception:
                st.dataframe(metrics_df.style.format(fmt), use_container_width=True)

    st.divider()

    # ────────────────────────────────────────────────────────────
    # 双栏：安全边际 + Agent 报告
    # ────────────────────────────────────────────────────────────
    left_col, right_col = st.columns([1, 2.8], gap="large")

    # ── 左栏：安全边际画布 ────────────────────────────────────────
    with left_col:
        st.markdown(f"### {T['mos_canvas']}")

        current_price   = dcf.get("current_price")
        intrinsic_value = dcf.get("intrinsic_value")
        mos             = dcf.get("margin_of_safety")
        dcf_err         = dcf.get("error")

        if dcf_err:
            st.warning(f"{T['dcf_err_label']}: {dcf_err}", icon="⚠️")
        else:
            # 两列价格卡 — st.columns 原生布局，彻底消灭折行与条纹错位
            _price_row(
                _fmt_price(current_price),
                _fmt_price(intrinsic_value),
                T["market_price"],
                T["intrinsic_label"],
            )

            if mos is not None:
                mos_color = ("#2da44e" if mos >= 10
                             else ("#9a6700" if mos >= -10 else "#cf222e"))
                # MoS 标签行 — CSS class 布局；st.progress 独占下一行
                st.markdown(
                    f'<div class="mos-row">'
                    f'<span class="mos-lbl">{T["mos_progress_lbl"]}</span>'
                    f'<span class="mos-val" style="color:{mos_color};">'
                    f'{_fmt_pct(mos)}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                # st.progress 独占整行 — 无嵌套列，CSS clear:both 隔离上方 div
                bar_val = min(max((mos + 60) / 120, 0.0), 1.0)
                st.progress(bar_val)

                if mos >= 30:
                    st.success(T["mos_deep"],    icon="📗")
                elif mos >= 10:
                    st.success(T["mos_slight"],  icon="📘")
                elif mos >= -10:
                    st.info(T["mos_fair"],       icon="📋")
                elif mos >= -30:
                    st.warning(T["mos_premium"], icon="⚠️")
                else:
                    st.error(T["mos_over"],      icon="🚨")

        st.divider()
        st.markdown(f'<p class="section-title">{T["dcf_params"]}</p>',
                    unsafe_allow_html=True)
        # 从 session_state 读取 Agent 预测增速（在 Agent Report 渲染后写入）
        _s1 = st.session_state.get("_stage1_growth")
        _s2 = st.session_state.get("_stage2_growth_start")
        # 若 Agent 已跑完，展示两阶段模式
        if _s1 is not None and _s2 is not None:
            st.markdown(
                f'<div class="two-stage-badge">{T["two_stage_badge"]}</div>',
                unsafe_allow_html=True,
            )
            _kpi(T["stage1_label"], f"{_s1*100:.1f}%")
            _kpi(T["stage2_label"], f"{_s2*100:.1f}%")
        else:
            # Agent 尚未运行，显示历史 FCF CAGR
            fcf_cagr_val = (f"{dcf.get('fcf_growth_rate', 0)*100:.1f}%"
                            if dcf.get("fcf_growth_rate") is not None else "N/A")
            _kpi(T["fcf_cagr_label"], fcf_cagr_val)
        _kpi(T["wacc_label"],   f"{dcf.get('discount_rate', 0.09)*100:.1f}%")
        _kpi(T["term_g_label"], f"{dcf.get('terminal_growth', 0.025)*100:.1f}%")

    # ── 右栏：Agent 思维链 ─────────────────────────────────────────
    with right_col:
        st.markdown(f"### {T['agent_panel']}")

        with st.spinner(T["broker_spin"]):
            broker_views, err_broker = load_broker_views(ticker_input, lang_code)

        tab_bull, tab_bear, tab_report = st.tabs([
            T["bull_tab"], T["bear_tab"], T["report_tab"],
        ])

        with tab_bull:
            if err_broker:
                st.warning(f"⚠️ {err_broker}", icon="🔌")
            if broker_views and broker_views.get("bullish"):
                consensus = broker_views.get("consensus", "N/A")
                st.markdown(
                    f'<div class="consensus-bar">📊 <b>{T["consensus_label"]}:</b> {consensus}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="signal-badge-bull">🟢 BULL CASE · 多方论据</div>',
                    unsafe_allow_html=True,
                )
                for pt in broker_views.get("bullish", []):
                    st.markdown(
                        f'<div class="bull-card">✅ &nbsp;{pt}</div>',
                        unsafe_allow_html=True,
                    )
            elif not err_broker:
                st.info(T["no_bull"])

        with tab_bear:
            if broker_views and broker_views.get("bearish"):
                st.markdown(
                    '<div class="signal-badge-bear">🔴 BEAR CASE · 空方论据</div>',
                    unsafe_allow_html=True,
                )
                for pt in broker_views.get("bearish", []):
                    st.markdown(
                        f'<div class="bear-card">⚠️ &nbsp;{pt}</div>',
                        unsafe_allow_html=True,
                    )
            elif not err_broker:
                st.info(T["no_bear"])

        with tab_report:
            with st.spinner(T["report_spin"]):
                m_json = metrics_df.to_json(orient="index") if (
                    metrics_df is not None and not metrics_df.empty) else "{}"
                d_json = json.dumps({
                    k: v for k, v in dcf.items()
                    if k != "ticker_obj"
                    and isinstance(v, (str, int, float, list, dict, type(None)))
                })
                final_report, stage1_g, stage2_g, updated_dcf, err_report = load_full_analysis(
                    ticker_input, m_json, d_json, lang_code)
                # 将 Agent 预测增速写入 session_state，供左侧面板展示
                st.session_state["_stage1_growth"]        = stage1_g
                st.session_state["_stage2_growth_start"]  = stage2_g
                st.session_state["_updated_dcf"]          = updated_dcf

            if err_report:
                st.error(f"{T['report_err']}: {err_report}")
            elif final_report:
                # 降级标题，包在 report-body div 内受 CSS 约束
                report_md = _downgrade_headers(final_report)
                st.markdown(
                    f'<div class="report-body">{report_md}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info(T["report_empty"])


# ══════════════════════════════════════════════════════════════════
# ── 历史盲测模式 ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════
else:
    # 公司名片（盲测模式同样展示）
    with st.spinner(f"🔍 {ticker_input}…"):
        company_info = load_company_info(ticker_input)
    st.markdown(f"### 🏢 {T['company_card_title']}")
    render_company_card(company_info, ticker_input, T)
    st.divider()

    with st.spinner(T["bt_spin"]):
        bt_result, bt_err = load_backtest(ticker_input, backtest_year)

    if bt_err:
        st.error(f"{T['bt_err']}: `{bt_err}`")
        st.stop()
    if bt_result is None:
        st.warning(T["bt_empty"])
        st.stop()

    # ── 决策横幅 ──────────────────────────────────────────────────
    decision = bt_result.get("decision", "HOLD")
    st.markdown(
        f"<div style='text-align:center;margin:16px 0 24px;"
        f"font-family:var(--mono);'>"
        f"<div style='font-size:0.70rem;letter-spacing:0.12em;text-transform:uppercase;"
        f"color:var(--text-secondary);margin-bottom:10px;'>{T['ai_decision']}</div>"
        f"{_decision_badge(decision)}</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── 历史量化仪表盘 ─────────────────────────────────────────────
    st.markdown(f"### {T['bt_snap']} — {backtest_year}")
    hist_metrics_df = bt_result.get("hist_metrics_df")
    hist_dcf        = bt_result.get("hist_dcf", {})

    if hist_metrics_df is not None and not hist_metrics_df.empty:
        roe_h  = _latest_metric(hist_metrics_df, "ROE (%)")
        roic_h = _latest_metric(hist_metrics_df, "ROIC (%)")
        gm_h   = _latest_metric(hist_metrics_df, "Gross Margin (%)")
        de_h   = _latest_metric(hist_metrics_df, "D/E Ratio")
        fcf_h  = _latest_metric(hist_metrics_df, "FCF ($M)")

        h1, h2, h3, h4, h5 = st.columns(5)
        h1.metric(T["roe_label"],  f"{roe_h:.1f}%"  if roe_h  is not None else "N/A")
        h2.metric(T["roic_label"], f"{roic_h:.1f}%" if roic_h is not None else "N/A")
        h3.metric(T["gm_label"],   f"{gm_h:.1f}%"   if gm_h   is not None else "N/A")
        h4.metric(T["de_label"],   f"{de_h:.2f}x"   if de_h   is not None else "N/A")
        with h5:
            fcf_h_str = (f"${fcf_h/1000:,.1f}B" if fcf_h and abs(fcf_h)>=1000
                         else (f"${fcf_h:,.0f}M" if fcf_h is not None else "N/A"))
            _kpi(T["fcf_label"], fcf_h_str)

        with st.expander(f"📅 {backtest_year} {T['trend_expand']}", expanded=False):
            _hfmt = {
                "ROE (%)": "{:.1f}%", "ROIC (%)": "{:.1f}%",
                "Gross Margin (%)": "{:.1f}%",
                "FCF ($M)": "{:,.0f}", "D/E Ratio": "{:.2f}x",
            }
            try:
                styled_h = hist_metrics_df.style.format(_hfmt).background_gradient(
                    cmap="RdYlGn", axis=None
                )
                st.dataframe(styled_h, use_container_width=True)
            except Exception:
                st.dataframe(hist_metrics_df.style.format(_hfmt), use_container_width=True)
    else:
        st.info(f"⚠️ {T['bt_insuf']}")

    st.divider()

    left_col, right_col = st.columns([1, 2.8], gap="large")

    with left_col:
        st.markdown(f"### {T['bt_mos']} — {backtest_year}")

        bp   = bt_result.get("backtest_price")
        cp   = bt_result.get("current_price")
        iv_h = hist_dcf.get("intrinsic_value")
        mos_h= hist_dcf.get("margin_of_safety")

        _price_row(
            _fmt_price(bp),
            _fmt_price(iv_h),
            f"{backtest_year} · {T['entry_price']}",
            T["hist_iv"],
        )

        if mos_h is not None:
            mos_color = "#2da44e" if mos_h >= 0 else "#cf222e"
            st.markdown(
                f'<div class="mos-row">'
                f'<span class="mos-lbl">{T["mos_progress_lbl"]}</span>'
                f'<span class="mos-val" style="color:{mos_color};">'
                f'{_fmt_pct(mos_h)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            bar_val = min(max((mos_h + 60) / 120, 0.0), 1.0)
            st.progress(bar_val)

        st.divider()

        # 收益对比表
        st.markdown(f"### {T['return_cmp']}")
        st.markdown(f"*{backtest_year} {T['held_note']}*")

        spy_bt  = bt_result.get("spy_backtest")
        spy_cur = bt_result.get("spy_current")
        t_ret   = bt_result.get("ticker_return")
        s_ret   = bt_result.get("spy_return")
        alpha   = bt_result.get("alpha")

        cmp_df = pd.DataFrame({
            T["asset"]:      [ticker_input, "SPY (S&P 500)"],
            T["entry_col"]:  [_fmt_price(bp), _fmt_price(spy_bt)],
            T["today_col"]:  [_fmt_price(cp), _fmt_price(spy_cur)],
            T["return_col"]: [_fmt_pct(t_ret), _fmt_pct(s_ret)],
        }).set_index(T["asset"])
        st.dataframe(cmp_df, use_container_width=True)

        if alpha is not None:
            sign  = "+" if alpha >= 0 else ""
            label = (T["outperform"] if alpha > 5
                     else (T["inline"] if alpha > -5 else T["underperform"]))
            a_color = ("var(--signal-bull)" if alpha > 5
                       else ("var(--signal-hold)" if alpha > -5 else "var(--signal-bear)"))
            st.markdown(
                f"<div style='text-align:center;margin-top:12px;"
                f"font-family:var(--mono);'>"
                f"<div style='font-size:0.68rem;letter-spacing:0.12em;text-transform:uppercase;"
                f"color:var(--text-secondary);margin-bottom:4px;'>Alpha vs SPY</div>"
                f"<div style='font-size:1.3rem;font-weight:700;color:{a_color};"
                f"white-space:nowrap;'>{sign}{alpha:.1f}%</div>"
                f"<div style='font-size:0.72rem;color:var(--text-secondary);"
                f"margin-top:2px;'>{label}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        if decision == "BUY":
            if t_ret is not None and t_ret > 0:
                st.success(f"{T['buy_correct']}\n\n{_fmt_pct(t_ret)}", icon="🏆")
            elif t_ret is not None:
                st.error(f"{T['buy_wrong']}\n\n{_fmt_pct(t_ret)}", icon="📉")
        elif decision == "SELL":
            if t_ret is not None and t_ret < 0:
                st.success(f"{T['sell_correct']}\n\n{_fmt_pct(t_ret)}", icon="🏆")
            else:
                st.warning(f"{T['sell_miss']}\n\n{_fmt_pct(t_ret)}", icon="📈")
        else:
            st.info(f"{T['hold_neutral']}\n\n{_fmt_pct(t_ret)}", icon="⚖️")

    with right_col:
        st.markdown(f"### {T['bt_agent_panel']} — {backtest_year}")

        hist_broker = bt_result.get("hist_broker", {})
        hist_report = bt_result.get("hist_report", "")

        tab_bull, tab_bear, tab_macro, tab_report_bt = st.tabs([
            T["bull_tab"], T["bear_tab"], T["macro_tab"], T["hist_report_tab"],
        ])

        with tab_bull:
            st.markdown(f"**{T['consensus_label']}:** `{hist_broker.get('consensus','N/A')}`")
            st.divider()
            for i, pt in enumerate(hist_broker.get("bullish", []), 1):
                st.markdown(f"**{i}.** {pt}")

        with tab_bear:
            for i, pt in enumerate(hist_broker.get("bearish", []), 1):
                st.markdown(f"**{i}.** {pt}")

        with tab_macro:
            st.markdown(
                f"**{T['macro_label']} ({backtest_year}):**\n\n"
                f"> {hist_broker.get('macro_context','N/A')}"
            )

        with tab_report_bt:
            if hist_report:
                st.markdown(
                    f'<div class="report-body">{_downgrade_headers(hist_report)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info(T["report_empty"])

        bt_md = bt_result.get("backtest_report", "")
        if bt_md:
            st.divider()
            st.download_button(
                label=T["download_btn"],
                data=bt_md,
                file_name=f"backtest_{ticker_input}_{backtest_year}.md",
                mime="text/markdown",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════
# 通用底部：【与投委会实时对话】 Interactive Chat
# ══════════════════════════════════════════════════════════════════
st.divider()
st.markdown(f"### {T['chat_title']}")
st.caption(T["chat_subtitle"])

# ── session_state 初始化 ──────────────────────────────────────────
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []   # list of {"role", "content"}
if "chat_ticker" not in st.session_state:
    st.session_state["chat_ticker"] = ""

# 当 ticker 切换时，清空对话历史并打招呼
_chat_ticker_now = st.session_state.get("_last_ticker", "")
if _chat_ticker_now and _chat_ticker_now != st.session_state.get("chat_ticker", ""):
    st.session_state["chat_messages"] = []
    st.session_state["chat_ticker"]   = _chat_ticker_now
    # 写入欢迎语（assistant 角色）
    welcome = T["chat_welcome"].replace("{ticker}", _chat_ticker_now)
    st.session_state["chat_messages"].append(
        {"role": "assistant", "content": welcome}
    )

# ── 判断是否已有分析数据（只有跑了分析才有意义开聊）─────────────────
_has_analysis = bool(
    run_btn
    and ticker_input
    and not st.session_state.get("_stage1_growth") is None
    or (run_btn and ticker_input and st.session_state.get("_updated_dcf"))
)

# 即使 Agent 还未跑完，只要有 ticker 数据就允许对话
_chat_ready = (run_btn or _analysis_done) and ticker_input

if not _chat_ready:
    st.info(T["chat_no_data"], icon="💡")
else:
    # ── 渲染历史消息 ──────────────────────────────────────────────
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"],
                             avatar="🤖" if msg["role"] == "assistant" else "🧑‍💼"):
            st.markdown(msg["content"])

    # ── 输入框 ────────────────────────────────────────────────────
    if user_input := st.chat_input(T["chat_placeholder"]):
        # 立即展示用户消息
        st.session_state["chat_messages"].append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(user_input)

        # 构建上下文（取已有的指标 / DCF / 报告）
        _m_json_chat = "{}"
        _d_json_chat = "{}"
        _report_chat = ""
        try:
            if "metrics_df" in dir() and metrics_df is not None and not metrics_df.empty:
                _m_json_chat = metrics_df.to_json(orient="index")
        except Exception:
            pass
        try:
            _dcf_src = st.session_state.get("_updated_dcf") or (dcf if "dcf" in dir() else {})
            _d_json_chat = json.dumps({
                k: v for k, v in _dcf_src.items()
                if isinstance(v, (str, int, float, list, dict, type(None)))
            })
        except Exception:
            pass
        # 从最近一条 assistant 消息中取报告（排除欢迎语）
        for _m in reversed(st.session_state["chat_messages"]):
            if _m["role"] == "assistant" and len(_m["content"]) > 200:
                _report_chat = _m["content"]
                break

        # 调用 LLM 并流式展示
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner(T["chat_thinking"]):
                history_for_llm = [
                    m for m in st.session_state["chat_messages"][:-1]  # 去掉刚加入的 user
                    if m["role"] in ("user", "assistant")
                ]
                reply = _call_chat_answer(
                    user_query   = user_input,
                    ticker       = ticker_input,
                    metrics_json = _m_json_chat,
                    dcf_json     = _d_json_chat,
                    agent_report = _report_chat,
                    history      = history_for_llm,
                    lang_code    = lang_code,
                )
            st.markdown(reply)

        st.session_state["chat_messages"].append(
            {"role": "assistant", "content": reply}
        )

    # ── 清空对话按钮 ──────────────────────────────────────────────
    if st.session_state["chat_messages"]:
        if st.button("🗑️  清空对话 / Clear Chat", use_container_width=False):
            st.session_state["chat_messages"] = []
            st.rerun()

st.divider()
st.caption(T["footer"])
