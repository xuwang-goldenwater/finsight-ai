"""
financial_engine.py
===================
价值投资量化分析引擎 — 基于 yfinance + DCF 模型

用法:
    python financial_engine.py AAPL
    python financial_engine.py BRK-B --discount-rate 0.10 --terminal-growth 0.025
"""

import sys
import time
import random
import argparse
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# TLS 级别浏览器伪装 Session
# 优先用 curl_cffi（完整模拟 Chrome TLS 指纹）
# 降级到 requests + UA Header
# ─────────────────────────────────────────────

def _make_session():
    """
    构建 yfinance 兼容的 Session。

    curl_cffi 在 TLS 握手层面模拟 Chrome，可绕过 Yahoo Finance
    的 JA3/JA4 指纹检测；requests 只能伪造 Header，仍会被识别。

    安装: pip install curl_cffi
    """
    try:
        from curl_cffi import requests as cffi_requests
        session = cffi_requests.Session(impersonate="chrome")
        return session, "curl_cffi"
    except ImportError:
        pass

    # fallback：requests + 浏览器 Header（效果较弱）
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    })
    retry = Retry(total=4, backoff_factor=2,
                  status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session, "requests"

# ─────────────────────────────────────────────
# 常量默认值
# ─────────────────────────────────────────────
DEFAULT_DISCOUNT_RATE   = 0.09   # WACC / 折现率
DEFAULT_TERMINAL_GROWTH = 0.025  # 永续增长率
DEFAULT_FCF_CAP         = 0.15   # FCF 增速上限，防止过热预测
FORECAST_YEARS          = 10     # 预测未来年数


# ═══════════════════════════════════════════════════════════════════
# 一、数据拉取与基本面指标计算
# ═══════════════════════════════════════════════════════════════════

def safe_get(data: dict, *keys, default=None):
    """多级 key 安全取值，任意层为 None / KeyError 均返回 default。"""
    val = data
    for k in keys:
        try:
            val = val[k]
            if val is None:
                return default
        except (KeyError, TypeError, IndexError):
            return default
    return val if val is not None else default


def fetch_financials(ticker: str) -> dict:
    """
    拉取 yfinance 财务数据，返回原始 dict。
    包含：income_stmt, balance_sheet, cashflow, info, history,
          quarterly_income, quarterly_cashflow, quarterly_balance

    TTM 逻辑：
    - 若年报最新列的年份落后当前年 ≥1 年，自动尝试拼接季报
      计算 TTM（Trailing Twelve Months）数据，并记录真实截止日期。
    - latest_data_date 字段：格式 'YYYY-MM-DD'，告知下游数据截止点。

    yfinance 1.x 已内建 curl_cffi 支持（安装后自动启用），
    无需手动传入 session，传入反而会引发兼容错误。
    """
    import datetime as _dt
    tk = yf.Ticker(ticker)
    result = {
        "income_stmt":         None,
        "balance_sheet":       None,
        "cashflow":            None,
        "info":                {},
        "history":             None,
        "ticker_obj":          tk,
        # 季报（用于 TTM 合成）
        "quarterly_income":    None,
        "quarterly_cashflow":  None,
        "quarterly_balance":   None,
        # 数据真实截止日期（YYYY-MM-DD），由 _resolve_latest_date() 填入
        "latest_data_date":    None,
    }

    # 各数据块独立拉取，单块失败不影响其他
    # yfinance 1.x 新属性名：income_stmt / cash_flow（旧名保留为 alias 但可能失效）
    def _get_income(t):
        df = getattr(t, "income_stmt", None)
        if df is None or (hasattr(df, "empty") and df.empty):
            df = getattr(t, "financials", None)
        return df

    def _get_cashflow(t):
        df = getattr(t, "cash_flow", None)
        if df is None or (hasattr(df, "empty") and df.empty):
            df = getattr(t, "cashflow", None)
        return df

    def _get_quarterly_income(t):
        df = getattr(t, "quarterly_income_stmt", None)
        if df is None or (hasattr(df, "empty") and df.empty):
            df = getattr(t, "quarterly_financials", None)
        return df

    def _get_quarterly_cashflow(t):
        df = getattr(t, "quarterly_cash_flow", None)
        if df is None or (hasattr(df, "empty") and df.empty):
            df = getattr(t, "quarterly_cashflow", None)
        return df

    def _get_info(t):
        """
        两路合并：fast_info 提供稳定的数字字段；
        tk.info 提供名称/行业/简介等文本字段，失败时静默忽略。
        """
        def _safe(obj, attr):
            try:
                return getattr(obj, attr)
            except Exception:
                return None

        merged = {}

        # ① fast_info — 价格/股本/52周高低等核心数字
        fi = getattr(t, "fast_info", None)
        if fi is not None:
            merged.update({
                "currentPrice":      _safe(fi, "last_price"),
                "sharesOutstanding": _safe(fi, "shares"),
                "marketCap":         _safe(fi, "market_cap"),
                "currency":          _safe(fi, "currency"),
                "fiftyTwoWeekHigh":  _safe(fi, "year_high"),
                "fiftyTwoWeekLow":   _safe(fi, "year_low"),
                "exchange":          _safe(fi, "exchange"),
            })

        # ② tk.info — 公司名/行业/简介/IPO 等文本字段（失败静默）
        try:
            full = t.info or {}
            for k in (
                "longName", "shortName", "longBusinessSummary",
                "sector", "industry", "country", "city",
                "fullTimeEmployees", "website",
                "firstTradeDateEpochUtc", "ipoDate",
                "fullExchangeName",
            ):
                if full.get(k) is not None:
                    merged[k] = full[k]
            # fast_info 未能拿到的数字字段从 info 补
            for k in ("currentPrice", "regularMarketPrice", "marketCap",
                      "sharesOutstanding", "fiftyTwoWeekHigh", "fiftyTwoWeekLow"):
                if not merged.get(k) and full.get(k):
                    merged[k] = full[k]
        except Exception:
            pass

        return merged if merged else {}

    def _get_history(t):
        # period="5d" 确保盘后/周末也能拿到最近一根有效 K 线
        hist = t.history(period="5d")
        return hist

    _pulls = [
        ("income_stmt",        lambda: _get_income(tk)),
        ("balance_sheet",      lambda: tk.balance_sheet),
        ("cashflow",           lambda: _get_cashflow(tk)),
        ("info",               lambda: _get_info(tk)),
        ("history",            lambda: _get_history(tk)),
        ("quarterly_income",   lambda: _get_quarterly_income(tk)),
        ("quarterly_cashflow", lambda: _get_quarterly_cashflow(tk)),
        ("quarterly_balance",  lambda: getattr(tk, "quarterly_balance_sheet", None)),
    ]
    for key, fn in _pulls:
        for attempt in range(3):
            try:
                result[key] = fn()
                break
            except Exception as e:
                msg = str(e)
                if "429" in msg or "Rate limit" in msg.lower() or "Too Many" in msg:
                    wait = 2 ** attempt + random.uniform(0.5, 1.5)
                    print(f"[限速] {key} 被限流，{wait:.1f}s 后重试（第 {attempt+1} 次）…")
                    time.sleep(wait)
                else:
                    print(f"[警告] {key} 拉取失败: {e}")
                    break
        # 各块之间随机短暂间隔，降低触发频率
        time.sleep(random.uniform(0.3, 0.8))

    # ── 解析真实数据截止日期，并在需要时合成 TTM 列 ──────────────
    result["latest_data_date"] = _resolve_latest_date(result)

    return result


def _col_date(df: "pd.DataFrame") -> "pd.Timestamp | None":
    """返回 DataFrame 列中最新的一个日期（Timestamp），列可能是 DatetimeIndex 或字符串。"""
    if df is None or df.empty:
        return None
    try:
        cols = pd.to_datetime(df.columns, errors="coerce")
        valid = cols.dropna()
        return valid.max() if not valid.empty else None
    except Exception:
        return None


def _resolve_latest_date(data: dict) -> str:
    """
    解析本次拉取的真实数据截止日期：

    策略：
    1. 用年报最新列日期作为基准。
    2. 若年报最新列落后当前年 ≥1 年，且季报有更新的数据，
       则用季报最新列日期作为截止日期，并把 TTM 合成列注入
       income_stmt / cashflow / balance_sheet（列名为 'TTM'）。
    3. 返回 'YYYY-MM-DD' 字符串，无法解析则返回 'unknown'。
    """
    import datetime as _dt
    today = _dt.date.today()
    current_year = today.year

    annual_date  = _col_date(data.get("income_stmt"))
    q_inc_date   = _col_date(data.get("quarterly_income"))
    q_cf_date    = _col_date(data.get("quarterly_cashflow"))
    q_bs_date    = _col_date(data.get("quarterly_balance"))

    # 过滤掉 None 和 NaT，保留有效 Timestamp
    def _valid_ts(d):
        try:
            return d is not None and not pd.isnull(d)
        except Exception:
            return False

    quarterly_dates = [d for d in [q_inc_date, q_cf_date, q_bs_date] if _valid_ts(d)]
    latest_q_date   = max(quarterly_dates) if quarterly_dates else None

    # 基准截止日期：默认用年报
    cutoff_date = annual_date

    # 如果年报落后 ≥1 年，且季报更新 → 合成 TTM 并更新截止日期
    if (_valid_ts(annual_date)
            and _valid_ts(latest_q_date)
            and annual_date.year < current_year
            and latest_q_date > annual_date):

        cutoff_date = latest_q_date

        for df_key, q_key in [
            ("income_stmt",  "quarterly_income"),
            ("cashflow",     "quarterly_cashflow"),
            ("balance_sheet","quarterly_balance"),
        ]:
            annual_df    = data.get(df_key)
            quarterly_df = data.get(q_key)
            ttm_col = _build_ttm_column(annual_df, quarterly_df)
            if ttm_col is not None and annual_df is not None and not annual_df.empty:
                ttm_label = f"TTM ({str(cutoff_date)[:10]})"
                # 插入 TTM 列作为最新列（最左列）
                try:
                    data[df_key] = pd.concat(
                        [annual_df, ttm_col.rename(ttm_label)], axis=1
                    )
                except Exception:
                    pass

    if cutoff_date is None:
        # 最后兜底：用季报日期
        cutoff_date = latest_q_date

    if cutoff_date is None:
        return "unknown"

    return str(cutoff_date)[:10]


def _build_ttm_column(annual_df: "pd.DataFrame",
                       quarterly_df: "pd.DataFrame") -> "pd.Series | None":
    """
    用最近 4 个季度数据合成 TTM（流量类科目做加法，余额类取最新季度）。

    流量科目（利润表、现金流）：TTM = 最新 4 季之和。
    余额科目（资产负债表）：TTM = 最新季度值（快照，不加总）。

    返回 pd.Series（index = 科目名），失败返回 None。
    """
    if quarterly_df is None or quarterly_df.empty:
        return None
    try:
        # 季报列按时间升序排列，取最右边 4 列
        q = quarterly_df.sort_index(axis=1, ascending=True)
        n = min(4, q.shape[1])
        recent_4q = q.iloc[:, -n:]
        # 判断是否为资产负债表（列数多，科目中含 Asset / Equity 等关键词）
        idx_str = " ".join(str(i).lower() for i in quarterly_df.index)
        is_balance = any(k in idx_str for k in
                         ("asset", "equity", "liabilit", "stockholder", "debt"))
        if is_balance:
            ttm = recent_4q.iloc[:, -1]   # 余额取最新季度快照
        else:
            ttm = recent_4q.sum(axis=1)   # 流量做 4Q 加总
        ttm = pd.to_numeric(ttm, errors="coerce")
        return ttm
    except Exception:
        return None


def _row(df: pd.DataFrame, *candidates) -> pd.Series:
    """
    在 DataFrame（列=年份）中按候选行名依次尝试，返回第一个找到的 Series。
    找不到则返回全 NaN Series。
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for name in candidates:
        # 精确匹配
        if name in df.index:
            return df.loc[name].apply(pd.to_numeric, errors="coerce")
        # 模糊匹配（忽略大小写 & 空格）
        matched = [i for i in df.index if name.lower().replace(" ", "") in i.lower().replace(" ", "")]
        if matched:
            return df.loc[matched[0]].apply(pd.to_numeric, errors="coerce")
    return pd.Series(dtype=float)


def calculate_metrics(data: dict) -> pd.DataFrame:
    """
    计算 5 个核心价值投资指标，返回按年份/TTM 索引的 DataFrame。

    若 fetch_financials() 已合成 TTM 列，该列会出现在 DataFrame 最右侧，
    行索引形如 'TTM (2025-09-30)'，供下游识别数据新鲜度。

    指标:
        ROE          净资产收益率  = Net Income / Stockholders' Equity
        ROIC         投入资本回报率 = NOPAT / Invested Capital
        Gross Margin 毛利率        = Gross Profit / Revenue
        FCF          自由现金流    = Operating CF - CapEx
        D/E Ratio    资产负债率    = Total Debt / Stockholders' Equity
    """
    inc = data["income_stmt"]
    bs  = data["balance_sheet"]
    cf  = data["cashflow"]

    # ── 利润表 ───────────────────────────────
    # yfinance 1.x 行名为 PascalCase；0.2.x 为带空格形式，两套都列出
    net_income    = _row(inc, "NetIncome", "Net Income")
    gross_profit  = _row(inc, "GrossProfit", "Gross Profit")
    total_revenue = _row(inc, "TotalRevenue", "Total Revenue", "Revenue")
    ebit          = _row(inc, "EBIT", "OperatingIncome", "Operating Income",
                         "EbitdaAndDepreciationAndAmortization")
    tax_row       = _row(inc, "TaxProvision", "Tax Provision", "IncomeTaxExpense",
                         "Income Tax Expense")

    # ── 资产负债表 ────────────────────────────
    equity       = _row(bs, "StockholdersEquity", "Stockholders Equity",
                        "CommonStockEquity", "TotalEquityGrossMinorityInterest",
                        "Total Stockholder Equity")
    total_debt   = _row(bs, "TotalDebt", "Total Debt", "LongTermDebt", "Long Term Debt")
    cash         = _row(bs, "CashAndCashEquivalents", "Cash And Cash Equivalents",
                        "CashCashEquivalentsAndShortTermInvestments",
                        "Cash And Cash Equivalents And Short Term Investments", "Cash")
    total_assets = _row(bs, "TotalAssets", "Total Assets")

    # ── 现金流量表 ────────────────────────────
    op_cf = _row(cf, "OperatingCashFlow", "Operating Cash Flow",
                 "CashFlowFromContinuingOperatingActivities",
                 "Total Cash From Operating Activities")
    capex = _row(cf, "CapitalExpenditure", "Capital Expenditure",
                 "CapitalExpenditures", "Capital Expenditures",
                 "PurchaseOfPropertyPlantAndEquipment")

    # ── 对齐年份 ──────────────────────────────
    all_series = [net_income, gross_profit, total_revenue, ebit,
                  equity, total_debt, cash, total_assets, tax_row, op_cf, capex]
    # 取所有非空 Series 的 index 交集（确保对齐）
    valid_indices = [s.index for s in all_series if not s.empty]
    if not valid_indices:
        return pd.DataFrame()

    common_cols = valid_indices[0]
    for idx in valid_indices[1:]:
        common_cols = common_cols.intersection(idx)

    if common_cols.empty:
        # fallback：使用 union，允许 NaN
        common_cols = valid_indices[0]
        for idx in valid_indices[1:]:
            common_cols = common_cols.union(idx)

    def align(s: pd.Series) -> pd.Series:
        return s.reindex(common_cols) if not s.empty else pd.Series(np.nan, index=common_cols)

    net_income    = align(net_income)
    gross_profit  = align(gross_profit)
    total_revenue = align(total_revenue)
    ebit          = align(ebit)
    equity        = align(equity)
    total_debt    = align(total_debt)
    cash          = align(cash)
    total_assets  = align(total_assets)
    tax_row       = align(tax_row)
    op_cf         = align(op_cf)
    capex         = align(capex)

    # ── 计算指标 ──────────────────────────────
    roe = (net_income / equity.replace(0, np.nan)) * 100

    # NOPAT = EBIT * (1 - 税率)；Invested Capital = Total Assets - Current Liabilities
    tax_rate = (tax_row / (ebit + 1e-9)).clip(0, 0.4)  # 粗估税率，限 0~40%
    nopat    = ebit * (1 - tax_rate)
    invested_capital = total_assets - (total_assets - equity - total_debt)  # ≈ equity + debt
    roic = (nopat / invested_capital.replace(0, np.nan)) * 100

    gross_margin = (gross_profit / total_revenue.replace(0, np.nan)) * 100

    # CapEx 在 yfinance 中通常为负值，取绝对值
    capex_abs = capex.abs()
    fcf = op_cf - capex_abs

    de_ratio = total_debt / equity.replace(0, np.nan)

    metrics = pd.DataFrame({
        "ROE (%)":          roe,
        "ROIC (%)":         roic,
        "Gross Margin (%)": gross_margin,
        "FCF ($M)":         fcf / 1e6,
        "D/E Ratio":        de_ratio,
    })

    # 按时间升序排列，取最近 5 年（含 TTM 列如果存在）
    # metrics 的 index 是日期（Timestamp 或 TTM 字符串）
    # TTM 行的 index 形如 'TTM (2025-09-30)'，不能与 Timestamp 混合排序
    str_idx    = [str(i) for i in metrics.index]
    ttm_mask   = [s.startswith("TTM") for s in str_idx]
    norm_mask  = [not b for b in ttm_mask]

    has_ttm = any(ttm_mask)

    if has_ttm:
        # 分离 TTM 行与普通年份行，各自操作后再拼接
        metrics_norm = metrics.iloc[[i for i, b in enumerate(norm_mask) if b]].copy()
        metrics_ttm  = metrics.iloc[[i for i, b in enumerate(ttm_mask)  if b]].copy()

        # 普通行：强制转换为 DatetimeIndex 再排序，避免混合类型比较
        try:
            metrics_norm.index = pd.to_datetime(metrics_norm.index, errors="coerce")
            metrics_norm = metrics_norm.sort_index(ascending=True).tail(4)
        except Exception:
            metrics_norm = metrics_norm.tail(4)

        metrics = pd.concat([metrics_norm, metrics_ttm])
    else:
        # 无 TTM：全部转为 DatetimeIndex 排序
        try:
            metrics.index = pd.to_datetime(metrics.index, errors="coerce")
            metrics = metrics.sort_index(ascending=True).tail(5)
        except Exception:
            metrics = metrics.tail(5)

    # 统一规范化为字符串索引：日期取前 10 字符，TTM 标签保留原样
    metrics.index = [
        str(d) if str(d).startswith("TTM") else str(d)[:10]
        for d in metrics.index
    ]
    return metrics


# ═══════════════════════════════════════════════════════════════════
# 二、DCF 估值模型
# ═══════════════════════════════════════════════════════════════════

def calculate_dcf_value(
    ticker: str,
    discount_rate: float   = DEFAULT_DISCOUNT_RATE,
    terminal_growth: float = DEFAULT_TERMINAL_GROWTH,
    fcf_cap: float         = DEFAULT_FCF_CAP,
    forecast_years: int    = FORECAST_YEARS,
    _data: dict            = None,   # 外部传入时复用，避免重复拉取
    stage1_growth: float   = None,   # Agent 预测：前 5 年增速（Two-Stage DCF）
    stage2_growth_start: float = None,  # Agent 预测：第 6 年起始过渡增速
) -> dict:
    """
    基于 DCF 模型计算内在价值（每股）。

    支持两种模式：
    • 单阶段（默认）：全程使用历史 FCF CAGR（上限 fcf_cap）
    • 两阶段（Agent 预测）：
        - 第 1-5 年：使用 stage1_growth（Agent 预测高速增长率）
        - 第 6-10 年：增速从 stage2_growth_start 线性递减至 terminal_growth

    公式:
        内在价值 = Σ [FCF_t / (1+r)^t]  +  Terminal Value / (1+r)^n
                   + Cash - Total Debt
        每股内在价值 = 内在价值 / 总股本

    返回 dict，包含所有中间变量供报告使用。
    """
    _two_stage = (stage1_growth is not None and stage2_growth_start is not None)

    result = {
        "ticker":               ticker.upper(),
        "intrinsic_value":      None,
        "current_price":        None,
        "margin_of_safety":     None,
        "fcf_growth_rate":      None,
        "base_fcf":             None,
        "discount_rate":        discount_rate,
        "terminal_growth":      terminal_growth,
        "stage1_growth":        stage1_growth,
        "stage2_growth_start":  stage2_growth_start,
        "two_stage_model":      _two_stage,
        "forecast_fcfs":        [],
        "pv_fcfs":              [],
        "terminal_value":       None,
        "pv_terminal":          None,
        "total_pv":             None,
        "cash":                 None,
        "total_debt":           None,
        "shares":               None,
        "latest_data_date":     None,   # 数据真实截止日期，由 fetch_financials 填入
        "error":                None,
    }

    try:
        data = _data if _data is not None else fetch_financials(ticker)
        tk   = data["ticker_obj"]

        # ── 数据截止日期（直接从 fetch_financials 结果中读取）────
        result["latest_data_date"] = data.get("latest_data_date")

        # ── 当前股价 ──────────────────────────
        hist = data["history"]
        if hist is not None and not hist.empty:
            result["current_price"] = float(hist["Close"].iloc[-1])
        else:
            result["current_price"] = safe_get(data["info"], "currentPrice", default=None)
            if result["current_price"] is None:
                result["current_price"] = safe_get(data["info"], "regularMarketPrice", default=None)

        # ── FCF 历史序列 ──────────────────────
        cf  = data["cashflow"]
        op_cf = _row(cf, "OperatingCashFlow", "Operating Cash Flow",
                     "CashFlowFromContinuingOperatingActivities",
                     "Total Cash From Operating Activities")
        capex = _row(cf, "CapitalExpenditure", "Capital Expenditure",
                     "CapitalExpenditures", "Capital Expenditures",
                     "PurchaseOfPropertyPlantAndEquipment")

        if op_cf.empty:
            result["error"] = "无法获取经营活动现金流数据"
            return result

        capex_abs = capex.abs() if not capex.empty else pd.Series(0, index=op_cf.index)
        fcf_series = (op_cf - capex_abs).sort_index(ascending=True)

        # 过滤极端异常值（绝对值为 0 或 NaN）
        fcf_series = fcf_series.replace(0, np.nan).dropna()
        if len(fcf_series) < 2:
            result["error"] = "FCF 历史数据不足，无法计算增速"
            return result

        # ── FCF 增速（CAGR）──────────────────
        # 使用最近 5 年（或全部可用年）计算 CAGR
        fcf_vals = fcf_series.tail(5).values.astype(float)
        n_years  = len(fcf_vals) - 1

        first_fcf = fcf_vals[0]
        last_fcf  = fcf_vals[-1]

        if first_fcf <= 0 or last_fcf <= 0:
            # 有负值时，改用算术平均年增量比率
            avg_fcf    = np.mean(np.abs(fcf_vals))
            raw_growth = np.mean(np.diff(fcf_vals)) / (avg_fcf + 1e-9)
        else:
            raw_growth = (last_fcf / first_fcf) ** (1 / n_years) - 1

        # 设定增速上限
        growth_rate = min(raw_growth, fcf_cap)
        growth_rate = max(growth_rate, -0.20)  # 下限 -20%，防止崩溃预测

        result["fcf_growth_rate"] = growth_rate
        result["base_fcf"]        = float(last_fcf)

        # ── 预测未来 10 年 FCF 并折现 ─────────
        # Two-Stage 模式：stage1_growth(yr1-5) + 线性过渡(yr6-10)
        # Single-Stage 模式：全程 growth_rate
        forecast_fcfs = []
        pv_fcfs       = []
        base = float(last_fcf)
        stage1 = stage1_growth if _two_stage else growth_rate
        # stage2 list: linear decline from stage2_growth_start → terminal_growth over yr6-10
        if _two_stage:
            n_transition = forecast_years - 5  # typically 5
            stage2_rates = [
                stage2_growth_start + (terminal_growth - stage2_growth_start) * i / max(n_transition - 1, 1)
                for i in range(n_transition)
            ]
        for t in range(1, forecast_years + 1):
            if _two_stage:
                g = stage1 if t <= 5 else stage2_rates[t - 6]
            else:
                g = growth_rate
            if t == 1:
                fcf_t = base * (1 + g)
            else:
                fcf_t = forecast_fcfs[-1] * (1 + g)
            pv_t   = fcf_t / (1 + discount_rate) ** t
            forecast_fcfs.append(fcf_t)
            pv_fcfs.append(pv_t)

        result["forecast_fcfs"] = forecast_fcfs
        result["pv_fcfs"]       = pv_fcfs

        # ── Terminal Value（Gordon Growth Model）─
        terminal_fcf   = forecast_fcfs[-1] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        pv_terminal    = terminal_value / (1 + discount_rate) ** forecast_years

        result["terminal_value"] = terminal_value
        result["pv_terminal"]    = pv_terminal

        total_pv = sum(pv_fcfs) + pv_terminal
        result["total_pv"] = total_pv

        # ── 资产负债表调整 ────────────────────
        bs = data["balance_sheet"]
        cash_row = _row(bs, "CashAndCashEquivalents", "Cash And Cash Equivalents",
                        "CashCashEquivalentsAndShortTermInvestments",
                        "Cash And Cash Equivalents And Short Term Investments", "Cash")
        debt_row = _row(bs, "TotalDebt", "Total Debt", "LongTermDebt", "Long Term Debt")

        latest_cash = float(cash_row.sort_index().iloc[-1]) if not cash_row.empty else 0.0
        latest_debt = float(debt_row.sort_index().iloc[-1]) if not debt_row.empty else 0.0
        # 有时 debt 已含负号，取绝对值
        latest_debt = abs(latest_debt)

        result["cash"]       = latest_cash
        result["total_debt"] = latest_debt

        # ── 总股本 ────────────────────────────
        shares = safe_get(data["info"], "sharesOutstanding", default=None)
        if shares is None:
            shares = safe_get(data["info"], "impliedSharesOutstanding", default=None)
        if shares is None:
            # fallback：尝试从市值 / 当前价格反推
            mktcap = safe_get(data["info"], "marketCap", default=None)
            price  = result["current_price"]
            if mktcap and price:
                shares = mktcap / price

        if not shares or shares <= 0:
            result["error"] = "无法获取总股本数据"
            return result

        result["shares"] = float(shares)

        # ── 内在价值（每股）─────────────────
        equity_value   = total_pv + latest_cash - latest_debt
        intrinsic_per_share = equity_value / float(shares)
        result["intrinsic_value"] = intrinsic_per_share

        # ── 安全边际 ──────────────────────────
        price = result["current_price"]
        if price and price > 0 and intrinsic_per_share:
            result["margin_of_safety"] = (intrinsic_per_share - price) / intrinsic_per_share * 100

    except Exception as e:
        result["error"] = f"DCF 计算异常: {e}"

    return result


# ═══════════════════════════════════════════════════════════════════
# 三、终端报告打印
# ═══════════════════════════════════════════════════════════════════

SEPARATOR = "═" * 62

def _fmt(val, fmt=".2f", prefix="", suffix="", na="N/A"):
    """安全格式化数值。"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return na
    return f"{prefix}{val:{fmt}}{suffix}"


def _print_company_profile(info: dict):
    """打印公司概况区块。"""
    if not info:
        print("  ⚠️  无法获取公司信息。")
        return

    name        = info.get("longName") or info.get("shortName", "N/A")
    sector      = info.get("sector",   "N/A")
    industry    = info.get("industry", "N/A")
    country     = info.get("country",  "N/A")
    exchange    = info.get("fullExchangeName") or info.get("exchange", "N/A")
    employees   = info.get("fullTimeEmployees")
    website     = info.get("website",  "N/A")
    currency    = info.get("currency", "USD")

    # IPO 日期
    ipo_str = "N/A"
    if info.get("ipoDate"):
        ipo_str = info["ipoDate"]
    elif info.get("firstTradeDateEpochUtc"):
        try:
            from datetime import datetime, timezone
            ipo_str = datetime.fromtimestamp(
                info["firstTradeDateEpochUtc"], tz=timezone.utc
            ).strftime("%Y-%m-%d")
        except Exception:
            pass

    # 价格与市值
    price    = info.get("currentPrice") or info.get("regularMarketPrice")
    mktcap   = info.get("marketCap")
    wk52h    = info.get("fiftyTwoWeekHigh")
    wk52l    = info.get("fiftyTwoWeekLow")

    price_str  = f"{currency} {price:,.2f}"   if price   else "N/A"
    mcap_str   = f"${mktcap/1e9:,.1f} B"      if mktcap  else "N/A"
    wk52_str   = (f"{currency} {wk52l:,.2f} ~ {wk52h:,.2f}"
                  if wk52h and wk52l else "N/A")
    emp_str    = f"{employees:,} 人"           if employees else "N/A"

    print(f"  公司全称:   {name}")
    print(f"  所属行业:   {sector}  /  {industry}")
    print(f"  注册地区:   {country}    上市交易所: {exchange}")
    print(f"  上市日期:   {ipo_str}    员工数: {emp_str}")
    print(f"  官网:       {website}")
    print()
    print(f"  当前股价:   {price_str}")
    print(f"  总市值:     {mcap_str}")
    print(f"  52 周区间:  {wk52_str}")

    # 业务简介（截取前 200 字）
    summary = info.get("longBusinessSummary", "")
    if summary:
        summary = summary.replace("\n", " ").strip()
        if len(summary) > 220:
            summary = summary[:220].rsplit(" ", 1)[0] + "…"
        print(f"\n  业务简介:   {summary}")


def print_report(ticker: str, metrics: pd.DataFrame, dcf: dict, info: dict = None):
    ticker_upper = ticker.upper()
    print(f"\n{SEPARATOR}")
    print(f"  📊  FinSight AI — 价值投资分析报告")
    print(f"  股票代码: {ticker_upper}")
    print(SEPARATOR)

    # ── 0. 公司概况 ────────────────────────────
    print("\n【零】公司概况\n")
    _print_company_profile(info or {})

    # ── 1. 基本面指标表 ────────────────────────
    print("\n【一】核心基本面指标（近 5 年）\n")
    if metrics.empty:
        print("  ⚠️  无法获取基本面数据。")
    else:
        # 格式化显示
        display = metrics.copy()
        for col in ["ROE (%)", "ROIC (%)", "Gross Margin (%)"]:
            if col in display.columns:
                display[col] = display[col].map(lambda x: f"{x:>7.1f}%" if pd.notna(x) else "   N/A")
        if "FCF ($M)" in display.columns:
            display["FCF ($M)"] = display["FCF ($M)"].map(
                lambda x: f"{x:>9,.0f}" if pd.notna(x) else "      N/A")
        if "D/E Ratio" in display.columns:
            display["D/E Ratio"] = display["D/E Ratio"].map(
                lambda x: f"{x:>7.2f}x" if pd.notna(x) else "   N/A")

        print(display.T.to_string())

    # ── 2. DCF 假设 ────────────────────────────
    print(f"\n{SEPARATOR}")
    print("\n【二】DCF 模型假设与预测\n")
    if dcf.get("error"):
        print(f"  ⚠️  DCF 计算失败: {dcf['error']}")
    else:
        g   = dcf["fcf_growth_rate"]
        r   = dcf["discount_rate"]
        tg  = dcf["terminal_growth"]
        base= dcf["base_fcf"]

        print(f"  基准 FCF（最近一年）:  {_fmt(base/1e9, '.3f', prefix='$', suffix=' B')}")
        print(f"  FCF 年均增速（含上限）: {_fmt(g*100, '.2f', suffix='%')}")
        print(f"  折现率（WACC）:         {r*100:.1f}%")
        print(f"  永续增长率:             {tg*100:.1f}%")
        print(f"  预测年限:               {FORECAST_YEARS} 年")
        print()

        # 预测现金流表
        print(f"  {'年份':>4}  {'预测 FCF ($B)':>14}  {'现值 PV ($B)':>13}")
        print(f"  {'─'*4}  {'─'*14}  {'─'*13}")
        for i, (f_val, pv_val) in enumerate(zip(dcf["forecast_fcfs"], dcf["pv_fcfs"]), 1):
            print(f"  {i:>4}  {f_val/1e9:>14,.3f}  {pv_val/1e9:>13,.3f}")

        tv  = dcf["terminal_value"]
        pvt = dcf["pv_terminal"]
        print(f"\n  永续价值（Terminal Value）: {_fmt(tv/1e9, '.3f', prefix='$', suffix=' B')}")
        print(f"  永续价值现值（PV）:          {_fmt(pvt/1e9, '.3f', prefix='$', suffix=' B')}")
        print(f"  FCF 现值总和:               {_fmt(sum(dcf['pv_fcfs'])/1e9, '.3f', prefix='$', suffix=' B')}")
        print(f"\n  持有现金:   {_fmt(dcf['cash']/1e9, '.3f', prefix='+$', suffix=' B')}")
        print(f"  总负债:     {_fmt(dcf['total_debt']/1e9, '.3f', prefix='-$', suffix=' B')}")
        shares_b = dcf["shares"] / 1e9 if dcf["shares"] else None
        print(f"  总股本:     {_fmt(shares_b, '.3f', suffix=' B 股')}")

    # ── 3. 估值结论 ────────────────────────────
    print(f"\n{SEPARATOR}")
    print("\n【三】估值结论\n")

    price = dcf.get("current_price")
    iv    = dcf.get("intrinsic_value")
    mos   = dcf.get("margin_of_safety")

    print(f"  当前市场股价:   {_fmt(price, '.2f', prefix='$')}")
    print(f"  AI 计算内在价值: {_fmt(iv, '.2f', prefix='$')}")
    print()

    if mos is not None:
        if mos >= 30:
            verdict = "✅  显著低估  — 具备充足安全边际，符合价值投资买入标准"
            color   = "▲"
        elif mos >= 10:
            verdict = "🟡  轻微低估  — 有一定安全边际，可考虑分批布局"
            color   = "△"
        elif mos >= -10:
            verdict = "⚪  基本合理  — 估值接近内在价值，建议继续观察"
            color   = "─"
        elif mos >= -30:
            verdict = "🟠  轻微高估  — 当前价格高于内在价值，需谨慎"
            color   = "▽"
        else:
            verdict = "🔴  显著高估  — 价格大幅超出内在价值，风险较高"
            color   = "▼"

        mos_sign = "+" if mos >= 0 else ""
        print(f"  安全边际（Margin of Safety）: {mos_sign}{mos:.1f}%")
        print(f"\n  投资结论: {verdict}")
    else:
        print("  ⚠️  无法计算安全边际（数据不足）。")

    print(f"\n{SEPARATOR}")
    print("  ⚠️  免责声明: 本报告仅供量化参考，不构成投资建议。")
    print(f"  请结合定性分析、行业背景与个人风险承受能力做出决策。")
    print(f"{SEPARATOR}\n")


# ═══════════════════════════════════════════════════════════════════
# 四、公开 API：供外部模块调用
# ═══════════════════════════════════════════════════════════════════

def analyze(
    ticker: str,
    discount_rate: float   = DEFAULT_DISCOUNT_RATE,
    terminal_growth: float = DEFAULT_TERMINAL_GROWTH,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    完整分析入口，返回 (metrics_df, dcf_result_dict, info_dict)。

    dcf_result_dict 包含 'latest_data_date' 字段（YYYY-MM-DD），
    标识本次量化数据的真实截止日期（可能为年报日期或 TTM 季报日期）。

    示例:
        from financial_engine import analyze
        metrics, dcf, info = analyze("AAPL")
        print(dcf["latest_data_date"])  # e.g. '2025-09-30'
    """
    # 只拉取一次，metrics 和 dcf 共享同一份数据，避免双倍请求
    data    = fetch_financials(ticker)
    metrics = calculate_metrics(data)
    dcf     = calculate_dcf_value(ticker, discount_rate, terminal_growth, _data=data)
    return metrics, dcf, data.get("info", {})


# ═══════════════════════════════════════════════════════════════════
# 五、CLI 入口
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="价值投资量化分析引擎 — FinSight AI"
    )
    parser.add_argument(
        "ticker",
        type=str,
        help="股票代码，例如 AAPL 或 BRK-B",
    )
    parser.add_argument(
        "--discount-rate",
        type=float,
        default=DEFAULT_DISCOUNT_RATE,
        help=f"折现率 / WACC（默认 {DEFAULT_DISCOUNT_RATE:.0%}）",
    )
    parser.add_argument(
        "--terminal-growth",
        type=float,
        default=DEFAULT_TERMINAL_GROWTH,
        help=f"永续增长率（默认 {DEFAULT_TERMINAL_GROWTH:.1%}）",
    )
    args = parser.parse_args()
    ticker = args.ticker.upper()

    # ── 公司确认：先拉 fast_info，展示名称让用户确认 ──────────────
    print(f"\n🔍  正在查找股票代码 {ticker}…")
    try:
        tk = yf.Ticker(ticker)
        fi = getattr(tk, "fast_info", None)
        company_name = None

        # fast_info 没有公司名，尝试从 info 或 get_info 拿
        try:
            info = tk.info or {}
            company_name = (
                info.get("longName")
                or info.get("shortName")
                or info.get("displayName")
            )
        except Exception:
            pass

        # 用收盘价做存在性验证
        last_price = getattr(fi, "last_price", None) if fi else None

        if not last_price and not company_name:
            print(f"  ⚠️  找不到 {ticker} 的有效数据，请检查股票代码是否正确。")
            print(f"      AppLovin → APP  |  苹果 → AAPL  |  英伟达 → NVDA")
            sys.exit(1)

        display = company_name or ticker
        price_str = f"${last_price:.2f}" if last_price else "价格暂不可用"
        print(f"\n  找到公司: 【{display}】({ticker})  当前价格: {price_str}")
        confirm = input("  确认是这家公司吗？按 Enter 继续，输入 n 退出: ").strip().lower()
        if confirm == "n":
            print("  已取消。请重新输入正确的股票代码。")
            sys.exit(0)

    except Exception as e:
        print(f"  ⚠️  代码验证失败: {e}，将直接继续分析…")

    print(f"\n⏳  正在获取 {ticker} 的财务数据，请稍候…")

    data    = fetch_financials(ticker)
    metrics = calculate_metrics(data)
    dcf     = calculate_dcf_value(
        ticker,
        discount_rate   = args.discount_rate,
        terminal_growth = args.terminal_growth,
        _data           = data,
    )
    print_report(ticker, metrics, dcf, info=data.get("info", {}))


if __name__ == "__main__":
    main()
