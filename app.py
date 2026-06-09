"""
app.py
======
FinSight AI — Streamlit 双语投资看板
Bilingual Value Investing Dashboard (简体中文 / English)

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
    page_title="FinSight AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# 全局 CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── 卡片容器 ── */
.metric-card {
    background: linear-gradient(135deg,#1e2130 0%,#252a3d 100%);
    border: 1px solid #2e3455;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 8px;
}

/* ── 决策徽章 ── */
.badge-buy  { background:#1a4731; color:#4ade80; border:1px solid #4ade80;
              border-radius:8px; padding:6px 16px; font-weight:700; font-size:1.05rem; }
.badge-hold { background:#3d3318; color:#facc15; border:1px solid #facc15;
              border-radius:8px; padding:6px 16px; font-weight:700; font-size:1.05rem; }
.badge-sell { background:#3d1a1a; color:#f87171; border:1px solid #f87171;
              border-radius:8px; padding:6px 16px; font-weight:700; font-size:1.05rem; }

/* ── 安全边际标签 ── */
.mos-label { font-size:0.82rem; color:#94a3b8; margin-bottom:4px; }

/* ── 章节小标题 ── */
.section-title {
    font-size:0.75rem; font-weight:600; letter-spacing:0.12em;
    text-transform:uppercase; color:#64748b; margin:12px 0 4px 0;
}

/* ── 关键数字：单行、不折行 ── */
.kpi-value {
    font-size: 1.15rem;
    font-weight: 700;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    color: #e2e8f0;
    line-height: 1.4;
}
.kpi-label {
    font-size: 0.72rem;
    color: #64748b;
    white-space: nowrap;
    margin-bottom: 2px;
}
.kpi-box {
    background: #1e2130;
    border: 1px solid #2e3455;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 6px;
}

/* ── 公司名片 ── */
.company-card {
    background: linear-gradient(135deg,#141928 0%,#1a1f35 100%);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.company-name {
    font-size: 1.35rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0 0 2px 0;
}
.company-meta {
    font-size: 0.8rem;
    color: #64748b;
}
.company-tag {
    display: inline-block;
    background: #1e3a5f;
    color: #93c5fd;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin: 4px 4px 0 0;
}
.price-big {
    font-size: 1.6rem;
    font-weight: 800;
    color: #4ade80;
    white-space: nowrap;
}

/* ── 强制 st.metric 不折行 ── */
[data-testid="stMetricValue"] {
    font-size: 1.05rem !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    white-space: nowrap !important;
}

/* ── 报告区大标题降级 ── */
.report-body h1 { font-size: 1.1rem !important; font-weight: 700; }
.report-body h2 { font-size: 1.0rem !important; font-weight: 700; }
.report-body h3 { font-size: 0.95rem !important; font-weight: 600; }
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
        "footer":            "🔬 **FinSight AI** · 由 yfinance + OpenAI GPT-4o-mini 驱动 · ⚠️ 仅供研究参考，不构成投资建议",
        # Outperform
        "outperform":        "跑赢 vs SPY",
        "inline":            "持平 vs SPY",
        "underperform":      "跑输 vs SPY",
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
        "footer":            "🔬 **FinSight AI** · Powered by yfinance & OpenAI GPT-4o-mini · ⚠️ For research only, not financial advice",
        "outperform":        "Outperformed SPY",
        "inline":            "In-line with SPY",
        "underperform":      "Underperformed SPY",
    },
}


# ══════════════════════════════════════════════════════════════════
# 侧边栏（语言切换必须最先渲染）
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── 语言切换（最顶部）──────────────────────────────────────────
    lang = st.radio(
        "🌐 Language / 语言",
        options=["简体中文", "English"],
        horizontal=True,
        key="lang",
    )
    T = _I18N[lang]             # 全局文本引用
    lang_code = "en" if lang == "English" else "zh"   # 传给 LLM

    st.markdown("## 📊 FinSight AI")
    st.markdown("**价值投资看板 · Value Investing Dashboard**")
    st.divider()

    ticker_input = st.text_input(
        T["ticker_label"],
        value="AAPL",
        placeholder="e.g. AAPL, MSFT, NVDA, APP",
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
        return None, err
    try:
        report = aa.run_full_analysis(
            ticker,
            json.loads(metrics_json),
            json.loads(dcf_json),
            use_search=False,
            save_report=False,
            lang=lang_code,
        )
        return report, None
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner=False, ttl=3600)
def load_backtest(ticker: str, backtest_year: int):
    _, _, bt, err = _import_backends()
    if err:
        return None, err
    try:
        return bt.run_backtest(ticker, backtest_year=backtest_year, save_report=False), None
    except Exception as e:
        return None, str(e)


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
    """单行不折行关键数字卡片（替代 st.metric 大字号）。"""
    html = (f'<div class="kpi-box">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'</div>')
    target = col if col else st
    target.markdown(html, unsafe_allow_html=True)

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
    st.markdown(f"""
<div class="company-card">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:8px;">
    <div>
      <div class="company-name">{name}</div>
      <div class="company-meta">{ticker} &nbsp;·&nbsp; {exchange}</div>
      <div style="margin-top:6px;">{tags_html}</div>
    </div>
    <div style="text-align:right;">
      <div class="price-big">{price_str}</div>
      <div style="font-size:0.75rem; color:#64748b;">{T['market_cap']}: {mcap_str}</div>
      <div style="font-size:0.75rem; color:#64748b; white-space:nowrap;">
        {T['week52']}: {wk52_str}
      </div>
    </div>
  </div>
  <hr style="border-color:#2e3455; margin:12px 0;">
  <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:6px; font-size:0.78rem; color:#94a3b8;">
    <span>📅 <b>{T['ipo_date']}</b>: {ipo_str}</span>
    <span>👥 <b>{T['employees']}</b>: {emp_str}</span>
    {"<span>🌐 <b>" + T['website'] + "</b>: <a href='" + website + "' target='_blank' style='color:#93c5fd;'>" + website.replace("https://","").rstrip("/") + "</a></span>" if website else ""}
  </div>
</div>
""", unsafe_allow_html=True)

    # ── 业务简介（折叠）────────────────────────────────────────────
    if summary:
        with st.expander(T["summary_expand"], expanded=False):
            st.markdown(
                f"<div style='font-size:0.85rem; color:#cbd5e1; line-height:1.7;'>{summary}</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════
# 主面板标题
# ══════════════════════════════════════════════════════════════════
st.markdown(f"# 📊 FinSight AI &nbsp;·&nbsp; {ticker_input or '—'}")
if is_backtest:
    st.markdown(
        f"**{T['bt_mode_tag']}** &nbsp;—&nbsp; "
        f"{T['snapshot_label']}: **{backtest_year}**"
    )
else:
    st.markdown(
        f"**{T['live_mode_tag']}** &nbsp;—&nbsp; "
        f"{T['today_label']}: **{date.today().isoformat()}**"
    )
st.divider()

# ── 等待按钮 ─────────────────────────────────────────────────────
if not run_btn:
    st.info(T["prompt_info"], icon="💡")
    st.stop()

if not ticker_input:
    st.warning(T["no_ticker_warn"], icon="⚠️")
    st.stop()


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
            st.dataframe(
                metrics_df.style.format({
                    "ROE (%)": "{:.1f}%", "ROIC (%)": "{:.1f}%",
                    "Gross Margin (%)": "{:.1f}%",
                    "FCF ($M)": "{:,.0f}", "D/E Ratio": "{:.2f}x",
                }).background_gradient(cmap="RdYlGn", axis=None),
                use_container_width=True,
            )

    st.divider()

    # ────────────────────────────────────────────────────────────
    # 双栏：安全边际 + Agent 报告
    # ────────────────────────────────────────────────────────────
    left_col, right_col = st.columns([1, 1.6], gap="large")

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
            pa, pb = st.columns(2)
            # 用 _kpi 防折行
            _kpi(T["market_price"],    _fmt_price(current_price), pa)
            _kpi(T["intrinsic_label"], _fmt_price(intrinsic_value), pb)

            if mos is not None:
                mos_color = "#4ade80" if mos >= 10 else ("#facc15" if mos >= -10 else "#f87171")
                st.markdown(
                    f'<p class="mos-label">{T["mos_progress_lbl"]}: '
                    f'<b style="color:{mos_color}; font-size:1rem; white-space:nowrap;">'
                    f'{_fmt_pct(mos)}</b></p>',
                    unsafe_allow_html=True,
                )
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
        fcf_cagr_val = (f"{dcf.get('fcf_growth_rate', 0)*100:.1f}%"
                        if dcf.get("fcf_growth_rate") is not None else "N/A")
        # 单列叠放 — 避免 left_col 内三列太窄导致折行
        _kpi(T["wacc_label"],     f"{dcf.get('discount_rate', 0.09)*100:.1f}%")
        _kpi(T["term_g_label"],   f"{dcf.get('terminal_growth', 0.025)*100:.1f}%")
        _kpi(T["fcf_cagr_label"], fcf_cagr_val)

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
                st.markdown(f"**{T['consensus_label']}:** `{broker_views.get('consensus','N/A')}`")
                st.divider()
                for i, pt in enumerate(broker_views.get("bullish", []), 1):
                    st.markdown(f"**{i}.** {pt}")
            elif not err_broker:
                st.info(T["no_bull"])

        with tab_bear:
            if broker_views and broker_views.get("bearish"):
                for i, pt in enumerate(broker_views.get("bearish", []), 1):
                    st.markdown(f"**{i}.** {pt}")
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
                final_report, err_report = load_full_analysis(
                    ticker_input, m_json, d_json, lang_code)

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
        f"<div style='text-align:center; margin:12px 0 20px;'>"
        f"<span style='color:#94a3b8; font-size:0.88rem;'>{T['ai_decision']}</span>"
        f"<br><br>{_decision_badge(decision)}</div>",
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
            st.dataframe(
                hist_metrics_df.style.format({
                    "ROE (%)": "{:.1f}%", "ROIC (%)": "{:.1f}%",
                    "Gross Margin (%)": "{:.1f}%",
                    "FCF ($M)": "{:,.0f}", "D/E Ratio": "{:.2f}x",
                }).background_gradient(cmap="RdYlGn", axis=None),
                use_container_width=True,
            )
    else:
        st.info(f"⚠️ {T['bt_insuf']}")

    st.divider()

    left_col, right_col = st.columns([1, 1.6], gap="large")

    with left_col:
        st.markdown(f"### {T['bt_mos']} — {backtest_year}")

        bp   = bt_result.get("backtest_price")
        cp   = bt_result.get("current_price")
        iv_h = hist_dcf.get("intrinsic_value")
        mos_h= hist_dcf.get("margin_of_safety")

        pa, pb = st.columns(2)
        _kpi(f"{backtest_year} {T['entry_price']}", _fmt_price(bp), pa)
        _kpi(T["hist_iv"], _fmt_price(iv_h), pb)

        if mos_h is not None:
            mos_color = "#4ade80" if mos_h >= 0 else "#f87171"
            st.markdown(
                f'<p style="text-align:center; color:#94a3b8; font-size:0.85rem;">'
                f'{T["mos_progress_lbl"]} = '
                f'<b style="color:{mos_color}; white-space:nowrap;">{_fmt_pct(mos_h)}</b></p>',
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
            color = "#4ade80" if alpha > 5 else ("#facc15" if alpha > -5 else "#f87171")
            st.markdown(
                f"<div style='text-align:center; margin-top:8px;'>"
                f"Alpha = <b style='color:{color}; font-size:1.15rem; white-space:nowrap;'>"
                f"{sign}{alpha:.1f}%</b><br>"
                f"<span style='color:#94a3b8; font-size:0.82rem;'>{label}</span></div>",
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
# 通用底部
# ══════════════════════════════════════════════════════════════════
st.divider()
st.caption(T["footer"])
