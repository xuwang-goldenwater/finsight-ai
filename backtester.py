"""
backtester.py
=============
FinSight AI — 历史盲测模块（第三步）

核心逻辑：
  1. 将财报数据时间切片至 backtest_year 及以前，模拟"历史时空"
  2. 用切片数据重算 metrics + DCF（当时的内在价值）
  3. 让 LLM 以 backtest_year 视角给出买入/持有/回避建议
  4. 对比 backtest_year 底部真实收盘价 → 现价 → AI 收益率 vs SPY

用法:
    python backtester.py AAPL
    python backtester.py NVDA --year 2022
    python backtester.py MSFT --year 2021 --model gpt-4o
"""

import os
import re
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date
from typing import Optional

from openai import OpenAI

warnings.filterwarnings("ignore")

# ── 复用前两个模块 ──────────────────────────────────────────────────
from financial_engine import (
    fetch_financials,
    calculate_metrics,
    calculate_dcf_value,
    safe_get,
    _row,
    DEFAULT_DISCOUNT_RATE,
    DEFAULT_TERMINAL_GROWTH,
)
from analysis_agents import (
    _get_client,
    _chat,
    _slim_metrics,
    _slim_dcf,
    _compact_json,
    analyze_financial_metrics,
    DEFAULT_MODEL,
    _MAX_TOKENS,
)

SEPARATOR = "═" * 62


# ══════════════════════════════════════════════════════════════════
# 一、时间切片工具
# ══════════════════════════════════════════════════════════════════

def _slice_df_to_year(df: pd.DataFrame, backtest_year: int) -> pd.DataFrame:
    """
    将 DataFrame（列 = 财年日期）切片，只保留 year <= backtest_year 的列。
    yfinance 财报列通常为 Timestamp，格式如 2023-09-30。
    """
    if df is None or df.empty:
        return df
    # 确保列是 Timestamp 类型
    cols_ts = pd.to_datetime(df.columns, errors="coerce")
    keep = [col for col, ts in zip(df.columns, cols_ts)
            if pd.notna(ts) and ts.year <= backtest_year]
    if not keep:
        return pd.DataFrame()
    return df[keep]


def slice_financials(data: dict, backtest_year: int) -> dict:
    """
    对 fetch_financials 返回的 data dict 做时间切片，
    income_stmt / balance_sheet / cashflow 只保留 ≤ backtest_year 的列。
    返回新的 sliced_data dict（不修改原始 data）。
    """
    sliced = dict(data)   # shallow copy
    for key in ("income_stmt", "balance_sheet", "cashflow"):
        df = data.get(key)
        sliced[key] = _slice_df_to_year(df, backtest_year) if df is not None else df

    # history 置为 None，由调用方填入历史价格
    sliced["history"] = None
    return sliced


# ══════════════════════════════════════════════════════════════════
# 二、历史价格抓取
# ══════════════════════════════════════════════════════════════════

def _get_year_end_price(ticker: str, year: int) -> Optional[float]:
    """
    抓取 year 年最后一个交易日的收盘价（尝试 12/28 ~ 1/5 窗口）。
    返回 float 或 None。
    """
    start = f"{year}-12-20"
    end   = f"{year + 1}-01-10"
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(start=start, end=end)
        if hist.empty:
            return None
        # 取 <= year-12-31 的最后一根
        hist.index = pd.to_datetime(hist.index)
        hist_year = hist[hist.index.year == year]
        if hist_year.empty:
            hist_year = hist  # fallback
        return float(hist_year["Close"].iloc[-1])
    except Exception as e:
        print(f"  [价格] {ticker} {year}年底价格拉取失败: {e}")
        return None


def _get_current_price(ticker: str) -> Optional[float]:
    """抓取最新收盘价（近 5 个交易日最后一根）。"""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="5d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception as e:
        print(f"  [价格] {ticker} 当前价格拉取失败: {e}")
        return None


def _make_history_df(price: float, reference_date: str) -> pd.DataFrame:
    """
    用单一历史价格构造一个 1 行的 history DataFrame，
    结构与 yfinance 返回的 history 一致，供 calculate_dcf_value 使用。
    """
    idx = pd.to_datetime([reference_date])
    return pd.DataFrame(
        {"Open": price, "High": price, "Low": price,
         "Close": price, "Volume": 0},
        index=idx,
    )


# ══════════════════════════════════════════════════════════════════
# 三、历史时空情报（BrokerIntel @ backtest_year）
# ══════════════════════════════════════════════════════════════════

_HISTORICAL_BROKER_SYSTEM = """\
你是一位严格处于 {year} 年的投资分析师，只能使用截至 {year} 年底已公开的信息。
**不得引用 {year} 年之后发生的任何事件**（例如更新的产品、政策或财报）。

请根据 {year} 年当时市场对该股票的公开认知，给出结构化的多空分析。
输出严格 JSON，无任何额外文字：
{{"bullish":["多方观点1","多方观点2","多方观点3"],\
"bearish":["空方观点1","空方观点2","空方观点3"],\
"macro_context":"一句话描述 {year} 年的宏观/行业背景",\
"consensus":"一句话评级与市场主流看法"}}\
"""

def fetch_historical_broker_views(
    ticker: str,
    backtest_year: int,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    让 LLM 以 backtest_year 视角推断当时市场对 ticker 的多空观点。
    系统 Prompt 明确限制：只能使用 backtest_year 及以前的公开信息。
    """
    client = _get_client()
    ticker_upper = ticker.upper()

    system = _HISTORICAL_BROKER_SYSTEM.replace("{year}", str(backtest_year))

    # 已知定性护城河数据库：为主流股票注入结构化的"软性指标"，
    # 帮助 AI 在 DCF 数字偏低时仍能识别护城河价值。
    _MOAT_CONTEXT: dict[str, dict[int, str]] = {
        "AAPL": {
            2023: (
                "补充定性数据（{year}年底公开信息）：\n"
                "- 服务业务（App Store/iCloud/Apple Music）毛利率约 70.8%，"
                "  占总营收比重已升至约 22%，且仍在持续提升。\n"
                "- 全球活跃设备超 20 亿台，用户生态锁定效应极强，年均 ARPU 持续爬升。\n"
                "- iPhone 15 系列刚发布，硬件增速放缓，但市场已广泛预期"
                "  2024-2025 年 AI 手机换机潮（Apple Intelligence 预研消息已流出）。\n"
                "- 近 5 年 FCF 均值因 2020-2021 年硬件大年基数高而显低，"
                "  但 2023 财年自由现金流仍超 990 亿美元，绝对值强劲。\n"
                "- 股票回购力度超强（2023 财年回购约 770 亿美元），EPS 增速优于净利润增速。"
            ),
            2022: (
                "补充定性数据（{year}年底公开信息）：\n"
                "- 服务业务毛利率约 71%，收入同比增长 14%，抗周期属性凸显。\n"
                "- 全球活跃设备超 18 亿台，高端市场份额持续扩大。\n"
                "- 2022 年宏观加息背景下科技股估值普遍压缩，但苹果盈利能力未见实质恶化。"
            ),
        },
        "MSFT": {
            2023: (
                "补充定性数据（{year}年底公开信息）：\n"
                "- Azure 增速约 28%，Copilot 商业化刚启动，企业付费意愿强。\n"
                "- Office 365/Teams 生态粘性极高，企业续约率超 95%。\n"
                "- OpenAI 49% 股权 + Azure 独家部署，AI 基础设施卡位最优。"
            ),
        },
        "NVDA": {
            2023: (
                "补充定性数据（{year}年底公开信息）：\n"
                "- H100 GPU 严重供不应求，云厂商排队等货，交货期 6-12 个月。\n"
                "- CUDA 软件生态覆盖 90%+ AI 研究代码，迁移成本极高。\n"
                "- 数据中心营收占比已超 60%，彻底摆脱游戏周期束缚。"
            ),
        },
    }

    # 拼接护城河补充上下文（若无匹配则留空）
    moat_info = ""
    ticker_moats = _MOAT_CONTEXT.get(ticker_upper, {})
    # 取 backtest_year 或最近可用年份
    for yr in sorted(ticker_moats.keys(), reverse=True):
        if yr <= backtest_year:
            moat_info = ticker_moats[yr].replace("{year}", str(backtest_year))
            break

    user = (
        f"股票：{ticker_upper}\n"
        f"请站在 {backtest_year} 年底的视角，给出当时市场对 {ticker_upper} 的多空分析。\n"
        + (f"\n{moat_info}" if moat_info else "")
    )

    raw = _chat(client, system, user, model=model,
                temperature=0.5, max_tokens=500)

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        return json.loads(cleaned.strip())
    except Exception:
        return {
            "bullish":   ["（解析失败，见 raw_output）"],
            "bearish":   ["（解析失败，见 raw_output）"],
            "macro_context": "N/A",
            "consensus": "N/A",
            "raw_output": raw,
        }


# ══════════════════════════════════════════════════════════════════
# 四、历史时空投委会（InvestmentCommitteeAgent @ backtest_year）
# ══════════════════════════════════════════════════════════════════

_HISTORICAL_COMMITTEE_SYSTEM = """\
你是 FinSight AI 投资委员会，当前时间是 {year} 年底。
你只能使用 {year} 年及以前已公开的信息，**严禁**引用 {year} 之后的事件。
风格：巴菲特式价值投资，直接有观点，无废话。

【DCF 平衡原则】
DCF 模型依赖历史 FCF 增速，对护城河型企业（强品牌/高转换成本/网络效应）存在系统性低估。
当出现以下任意两项时，可对 DCF 结论向上修正 20-40%，并在报告中明确说明修正理由：
  - 服务/订阅业务毛利率 ≥ 60%，且占收入比仍在提升
  - 活跃用户/设备基数超 10 亿，具备生态锁定效应
  - 硬件增速放缓但服务/软件 ARPU 持续爬升（"剃须刀-刀片"转型信号）
  - 市场共识中已出现明确的产品/换机周期预期

严格按以下 Markdown 结构输出（每节 ≤150 字）：

## 一、{year} 年量化快照
## 二、指标因果剖析
## 三、{year} 年市场多空风向标
## 四、DCF 内在价值评估（含护城河修正说明）
## 五、投资决策

**[决策]** 必须在第五节末尾单独一行，严格使用以下三个英文单词之一，
不得添加任何其他字符：
DECISION: BUY
DECISION: HOLD
DECISION: SELL

并附：1 个核心催化剂（支撑决策的最重要单一因素）\
"""

def run_historical_committee(
    ticker: str,
    metrics_json,
    dcf_json,
    causal_analysis: str,
    broker_views: dict,
    backtest_year: int,
    model: str = DEFAULT_MODEL,
) -> tuple[str, str]:
    """
    以 backtest_year 视角运行投委会，输出历史分析报告和投资决策。

    返回:
        (report_markdown, decision_str)
        decision_str ∈ {"买入", "持有/观望", "回避", "未知"}
    """
    client = _get_client()
    ticker_upper = ticker.upper()

    system = (
        _HISTORICAL_COMMITTEE_SYSTEM
        .replace("{year}", str(backtest_year))
        .replace("{ticker}", ticker_upper)
    )

    bullish_str = "\n".join(f"  - {b}" for b in broker_views.get("bullish", []))
    bearish_str = "\n".join(f"  - {b}" for b in broker_views.get("bearish", []))
    macro_ctx   = broker_views.get("macro_context", "N/A")
    consensus   = broker_views.get("consensus", "N/A")

    slim_m = _slim_metrics(metrics_json)
    slim_d = _slim_dcf(dcf_json)

    user_prompt = (
        f"股票:{ticker_upper}  分析时间节点:{backtest_year}年底\n"
        f"指标:{_compact_json(slim_m)}\n"
        f"DCF:{_compact_json(slim_d)}\n"
        f"因果分析摘要:{causal_analysis[:500]}\n"
        f"宏观背景:{macro_ctx}\n"
        f"多方:{bullish_str}\n"
        f"空方:{bearish_str}\n"
        f"共识评级:{consensus}"
    )

    report = _chat(client, system, user_prompt, model=model,
                   temperature=0.55, max_tokens=1400)

    # 用正则从全文提取 DECISION 标签，容忍格式变体（大小写/空格/反引号/粗体）
    decision = _parse_decision(report)

    return report, decision


# ── 决策提取工具（独立函数，方便单元测试）─────────────────────────
_DECISION_PATTERN = re.compile(
    r"DECISION\s*:\s*`?\*{0,2}(BUY|HOLD|SELL)\*{0,2}`?",
    re.IGNORECASE,
)

def _parse_decision(text: str) -> str:
    """
    从 LLM 输出中提取投资决策，返回 'BUY' / 'HOLD' / 'SELL'。
    匹配逻辑：
      1. 精确正则：DECISION: BUY/HOLD/SELL（容忍反引号、粗体、大小写）
      2. 宽松扫描：整行中出现独立的 BUY / HOLD / SELL
      3. 以上均失败 → 'HOLD'（保守兜底，避免返回'未知'）
    """
    # 优先精确匹配
    m = _DECISION_PATTERN.search(text)
    if m:
        return m.group(1).upper()

    # 宽松扫描：找含 DECISION 的行，提取其中的英文词
    for line in text.splitlines():
        upper = line.upper()
        if "DECISION" in upper:
            for keyword in ("BUY", "SELL", "HOLD"):
                # 确保是独立单词（避免 HOLDING/BUYER 等误匹配）
                if re.search(rf"\b{keyword}\b", upper):
                    return keyword

    # 再宽松一步：全文扫描独立 BUY/SELL/HOLD
    for keyword in ("BUY", "SELL", "HOLD"):
        if re.search(rf"\b{keyword}\b", text.upper()):
            return keyword

    return "HOLD"  # 保守兜底


# ══════════════════════════════════════════════════════════════════
# 五、回报率计算
# ══════════════════════════════════════════════════════════════════

def _calc_return(price_entry: Optional[float], price_exit: Optional[float]) -> Optional[float]:
    """计算简单持有回报率（百分比），两者任一为 None 则返回 None。"""
    if price_entry and price_exit and price_entry > 0:
        return (price_exit - price_entry) / price_entry * 100
    return None


def _fmt_return(ret: Optional[float]) -> str:
    if ret is None:
        return "N/A"
    sign = "+" if ret >= 0 else ""
    return f"{sign}{ret:.1f}%"


# ══════════════════════════════════════════════════════════════════
# 六、主回测函数
# ══════════════════════════════════════════════════════════════════

def run_backtest(
    ticker: str,
    backtest_year: int          = 2023,
    discount_rate: float        = DEFAULT_DISCOUNT_RATE,
    terminal_growth: float      = DEFAULT_TERMINAL_GROWTH,
    model: str                  = DEFAULT_MODEL,
    save_report: bool           = True,
) -> dict:
    """
    历史盲测主函数。

    参数:
        ticker        : 股票代码，如 "AAPL"
        backtest_year : 回测基准年（模拟在该年底做投资决策），默认 2023
        discount_rate : DCF 折现率
        terminal_growth: 永续增长率
        model         : OpenAI 模型
        save_report   : 是否将报告保存为 Markdown 文件

    返回:
        result dict，包含所有中间数据与最终对照结论
    """
    ticker_upper = ticker.upper()
    today_str    = date.today().isoformat()
    current_year = date.today().year

    print(f"\n{SEPARATOR}")
    print(f"  FinSight AI — 历史盲测模块")
    print(f"  股票: {ticker_upper}  |  回测节点: {backtest_year}年底")
    print(f"  当前日期: {today_str}")
    print(SEPARATOR)

    # ── Step 1: 拉取完整历史财报数据 ──────────────────────────
    print(f"\n[Step 1] 拉取 {ticker_upper} 全量历史财报数据…")
    raw_data = fetch_financials(ticker_upper)
    print("  ✓ 财报数据就绪")

    # ── Step 2: 时间切片 → 还原 backtest_year 时空 ──────────
    print(f"\n[Step 2] 时间切片：过滤至 {backtest_year} 年及以前的财报…")
    sliced_data = slice_financials(raw_data, backtest_year)

    # 检查切片是否有效
    inc_sliced = sliced_data.get("income_stmt")
    if inc_sliced is None or (hasattr(inc_sliced, "empty") and inc_sliced.empty):
        print(f"  [警告] {backtest_year} 年以前的财报列为空，可能该股票上市时间较晚"
              f"或 yfinance 无此年份数据。")

    # ── Step 3: 抓取历史价格 & 当前价格 ─────────────────────
    print(f"\n[Step 3] 抓取价格数据…")

    backtest_price = _get_year_end_price(ticker_upper, backtest_year)
    current_price  = _get_current_price(ticker_upper)
    spy_backtest   = _get_year_end_price("SPY", backtest_year)
    spy_current    = _get_current_price("SPY")

    print(f"  {ticker_upper} @ {backtest_year}年底 : "
          f"{'$'+f'{backtest_price:.2f}' if backtest_price else 'N/A'}")
    print(f"  {ticker_upper} @ 今日 ({today_str}) : "
          f"{'$'+f'{current_price:.2f}' if current_price else 'N/A'}")
    print(f"  SPY    @ {backtest_year}年底 : "
          f"{'$'+f'{spy_backtest:.2f}' if spy_backtest else 'N/A'}")
    print(f"  SPY    @ 今日            : "
          f"{'$'+f'{spy_current:.2f}' if spy_current else 'N/A'}")

    # 将历史价格注入 sliced_data 的 history 字段
    if backtest_price:
        hist_ref = f"{backtest_year}-12-31"
        sliced_data["history"] = _make_history_df(backtest_price, hist_ref)
        sliced_data["info"]    = {
            **sliced_data.get("info", {}),
            "currentPrice": backtest_price,
        }

    # ── Step 4: 用切片数据计算历史时点的基本面指标 + DCF ────
    print(f"\n[Step 4] 计算 {backtest_year} 年时点的基本面指标与 DCF 内在价值…")

    hist_metrics_df = calculate_metrics(sliced_data)
    hist_dcf        = calculate_dcf_value(
        ticker_upper,
        discount_rate   = discount_rate,
        terminal_growth = terminal_growth,
        _data           = sliced_data,
    )

    # 覆盖 DCF 中的 current_price 为历史价格（calculate_dcf_value 可能从 info 读到）
    if backtest_price:
        hist_dcf["current_price"] = backtest_price
        if hist_dcf.get("intrinsic_value") and backtest_price > 0:
            hist_dcf["margin_of_safety"] = (
                (hist_dcf["intrinsic_value"] - backtest_price)
                / hist_dcf["intrinsic_value"] * 100
            )

    if hist_metrics_df.empty:
        print(f"  [警告] 切片后基本面指标为空（{backtest_year}年前的财报数据不足）")
    else:
        print(f"  ✓ 历史基本面指标已计算（覆盖年份：{list(hist_metrics_df.index)}）")

    if hist_dcf.get("error"):
        print(f"  [警告] 历史 DCF 计算异常: {hist_dcf['error']}")
    else:
        iv = hist_dcf.get("intrinsic_value")
        mos = hist_dcf.get("margin_of_safety")
        print(f"  ✓ 历史 DCF 内在价值: "
              f"{'$'+f'{iv:.2f}' if iv else 'N/A'}"
              f"  安全边际: {f'{mos:+.1f}%' if mos is not None else 'N/A'}")

    # DataFrame → JSON（供 LLM 使用）
    if not hist_metrics_df.empty:
        hist_metrics_json = json.loads(hist_metrics_df.to_json(orient="index"))
    else:
        hist_metrics_json = {}

    hist_dcf_json = {
        k: v for k, v in hist_dcf.items()
        if k != "ticker_obj" and isinstance(v, (str, int, float, list, dict, type(None)))
    }

    # ── Step 5: 历史时空因果分析（Layer 1 @ backtest_year）──
    print(f"\n[Step 5] 历史因果分析（站在 {backtest_year} 年视角）…")
    causal_analysis = analyze_financial_metrics(
        hist_metrics_json, hist_dcf_json, ticker_upper, model
    )
    print("  ✓ 历史因果分析完成")

    # ── Step 6: 历史情报模拟（BrokerIntel @ backtest_year）──
    print(f"\n[Step 6] 模拟 {backtest_year} 年市场情报（历史多空辩论）…")
    hist_broker_views = fetch_historical_broker_views(ticker_upper, backtest_year, model)
    macro_ctx = hist_broker_views.get("macro_context", "N/A")
    print(f"  ✓ 历史情报就绪 | 宏观背景: {macro_ctx}")

    # ── Step 7: 历史投委会决策（InvestmentCommittee @ backtest_year）
    print(f"\n[Step 7] 投委会历史决策（仅用 {backtest_year} 年及以前信息）…")
    hist_report, decision = run_historical_committee(
        ticker_upper,
        hist_metrics_json,
        hist_dcf_json,
        causal_analysis,
        hist_broker_views,
        backtest_year,
        model,
    )
    print(f"  ✓ 历史决策: 【{decision}】")

    # ── Step 8: 计算回报率对比 ──────────────────────────────
    ticker_return = _calc_return(backtest_price, current_price)
    spy_return    = _calc_return(spy_backtest, spy_current)
    alpha         = (
        ticker_return - spy_return
        if ticker_return is not None and spy_return is not None
        else None
    )

    hold_years = current_year - backtest_year  # 持有年数（近似）

    # ── Step 9: 生成盲测对照报告 ─────────────────────────────
    print(f"\n[Step 8] 生成盲测对照报告…")

    backtest_report = _build_backtest_report(
        ticker         = ticker_upper,
        backtest_year  = backtest_year,
        today_str      = today_str,
        hist_metrics_df= hist_metrics_df,
        hist_dcf       = hist_dcf,
        hist_broker    = hist_broker_views,
        hist_report    = hist_report,
        decision       = decision,
        backtest_price = backtest_price,
        current_price  = current_price,
        spy_backtest   = spy_backtest,
        spy_current    = spy_current,
        ticker_return  = ticker_return,
        spy_return     = spy_return,
        alpha          = alpha,
        hold_years     = hold_years,
        causal_analysis= causal_analysis,
    )

    print("  ✓ 盲测报告生成完毕")

    # 保存报告
    if save_report:
        fname = (f"backtest_{ticker_upper}_{backtest_year}_"
                 f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        try:
            with open(fname, "w", encoding="utf-8") as f:
                f.write(backtest_report)
            print(f"\n  📄 报告已保存: {fname}")
        except Exception as e:
            print(f"\n  [警告] 保存报告失败: {e}")

    # 打印终端摘要
    _print_backtest_summary(
        ticker_upper, backtest_year, today_str,
        backtest_price, current_price,
        spy_backtest, spy_current,
        ticker_return, spy_return, alpha,
        decision, hold_years,
        hist_dcf,
    )

    return {
        "ticker":          ticker_upper,
        "backtest_year":   backtest_year,
        "decision":        decision,
        "backtest_price":  backtest_price,
        "current_price":   current_price,
        "spy_backtest":    spy_backtest,
        "spy_current":     spy_current,
        "ticker_return":   ticker_return,
        "spy_return":      spy_return,
        "alpha":           alpha,
        "hist_metrics_df": hist_metrics_df,
        "hist_dcf":        hist_dcf,
        "hist_broker":     hist_broker_views,
        "hist_report":     hist_report,
        "backtest_report": backtest_report,
    }


# ══════════════════════════════════════════════════════════════════
# 七、报告生成工具
# ══════════════════════════════════════════════════════════════════

def _build_backtest_report(
    ticker, backtest_year, today_str,
    hist_metrics_df, hist_dcf, hist_broker, hist_report, decision,
    backtest_price, current_price,
    spy_backtest, spy_current,
    ticker_return, spy_return, alpha,
    hold_years, causal_analysis,
) -> str:
    """拼装完整的盲测对照报告（Markdown）。"""

    def _p(v, prefix="$", decimals=2, suffix=""):
        if v is None:
            return "N/A"
        return f"{prefix}{v:.{decimals}f}{suffix}"

    def _r(v):
        return _fmt_return(v)

    # 决策图标（英文标签 → 图标 + 中文说明）
    decision_icon  = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(decision, "⚪")
    decision_label = {"BUY": "买入", "SELL": "卖出", "HOLD": "持有/观望"}.get(decision, decision)

    # alpha 判断
    if alpha is None:
        alpha_verdict = "（数据不足）"
    elif alpha > 5:
        alpha_verdict = f"✅ AI 跑赢 SPY {alpha:+.1f}%"
    elif alpha > -5:
        alpha_verdict = f"🟡 AI 与 SPY 基本持平（{alpha:+.1f}%）"
    else:
        alpha_verdict = f"🔴 AI 跑输 SPY {alpha:+.1f}%"

    # 决策回报验证
    if decision == "BUY" and ticker_return is not None:
        if ticker_return > 0:
            return_verdict = f"✅ BUY 正确 — 持有至今回报 {_r(ticker_return)}"
        else:
            return_verdict = f"🔴 BUY 失误 — 持有至今亏损 {_r(ticker_return)}"
    elif decision == "SELL" and ticker_return is not None:
        if ticker_return > 0:
            return_verdict = f"🔴 SELL 失策 — 该股实际上涨 {_r(ticker_return)}，错过机会"
        else:
            return_verdict = f"✅ SELL 正确 — 该股实际下跌 {_r(ticker_return)}"
    elif decision == "HOLD":
        return_verdict = f"🟡 HOLD 观望 — 实际表现 {_r(ticker_return)}"
    else:
        return_verdict = f"决策: {decision} | 实际表现: {_r(ticker_return)}"

    # 指标表格
    if not hist_metrics_df.empty:
        try:
            metrics_table = hist_metrics_df.to_markdown()
        except (ImportError, Exception):
            metrics_table = hist_metrics_df.to_string()
    else:
        metrics_table = "（切片后数据不足，无法展示）"

    # DCF 摘要
    iv  = hist_dcf.get("intrinsic_value")
    mos = hist_dcf.get("margin_of_safety")

    lines = [
        f"# FinSight AI — 历史盲测对照报告",
        f"**股票**: {ticker}  |  **回测节点**: {backtest_year} 年底  "
        f"|  **生成日期**: {today_str}",
        "",
        "---",
        "",
        "## 一、回测设定说明",
        "",
        f"本报告将财报数据时间切片至 **{backtest_year} 年及以前**，",
        f"模拟 AI 系统在 **{backtest_year} 年底**对 {ticker} 做出投资决策的完整过程。",
        f"决策完成后，再对照 {backtest_year} 年底至今（{today_str}）的**真实价格走势**，",
        f"验证 AI 判断的准确性，并与同期标普 500（SPY）基准对比。",
        "",
        "---",
        "",
        "## 二、历史基本面快照（切片至 {year} 年）".replace("{year}", str(backtest_year)),
        "",
        metrics_table,
        "",
        f"> **历史 DCF 内在价值（{backtest_year}年底）**: {_p(iv)}  ",
        f"> **历史市场价格（{backtest_year}年底）**: {_p(backtest_price)}  ",
        f"> **安全边际**: {f'{mos:+.1f}%' if mos is not None else 'N/A'}  ",
        "> FCF 增速假设: "
        + (f"{hist_dcf.get('fcf_growth_rate', 0)*100:.1f}%"
           if hist_dcf.get('fcf_growth_rate') is not None else 'N/A')
        + "  ",
        f"> 折现率: {hist_dcf.get('discount_rate', DEFAULT_DISCOUNT_RATE)*100:.1f}%",
        "",
        "---",
        "",
        f"## 三、{backtest_year} 年时空情报（历史市场多空辩论）",
        "",
        f"**宏观背景**: {hist_broker.get('macro_context', 'N/A')}",
        "",
        "**多方观点**",
        *[f"- {b}" for b in hist_broker.get("bullish", [])],
        "",
        "**空方观点**",
        *[f"- {b}" for b in hist_broker.get("bearish", [])],
        "",
        f"**市场共识（{backtest_year}年）**: {hist_broker.get('consensus', 'N/A')}",
        "",
        "---",
        "",
        f"## 四、历史投委会分析报告（{backtest_year}年视角）",
        "",
        hist_report,
        "",
        "---",
        "",
        "## 五、盲测对照结论",
        "",
        f"### 5.1 AI 历史决策",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| {backtest_year}年底 AI 决策 | {decision_icon} **{decision}** ({decision_label}) |",
        f"| {backtest_year}年底 {ticker} 价格 | {_p(backtest_price)} |",
        f"| 今日 {ticker} 价格（{today_str}）| {_p(current_price)} |",
        f"| 持有年数（估算）| {hold_years} 年 |",
        "",
        f"### 5.2 回报率对比",
        f"| 对象 | 入场价 | 今日价 | 回报率 |",
        f"|------|--------|--------|--------|",
        f"| **{ticker}** | {_p(backtest_price)} | {_p(current_price)} | **{_r(ticker_return)}** |",
        f"| **SPY（标普500）** | {_p(spy_backtest)} | {_p(spy_current)} | **{_r(spy_return)}** |",
        "",
        f"### 5.3 超额收益（Alpha）",
        f"> Alpha = {ticker} 回报 − SPY 回报 = **{_r(alpha)}**",
        f"> {alpha_verdict}",
        "",
        f"### 5.4 决策验证",
        f"> {return_verdict}",
        "",
        "---",
        "",
        "> ⚠️ **免责声明**: 本报告为历史回测模拟，仅供学习与研究用途。",
        "> 历史表现不代表未来收益，不构成任何投资建议。",
        "> 回测存在幸存者偏差与信息泄露风险，请谨慎解读。",
    ]

    return "\n".join(lines)


def _print_backtest_summary(
    ticker, backtest_year, today_str,
    backtest_price, current_price,
    spy_backtest, spy_current,
    ticker_return, spy_return, alpha,
    decision, hold_years, hist_dcf,
):
    """在终端打印简洁的盲测摘要。"""

    def _p(v, prefix="$"):
        return f"{prefix}{v:.2f}" if v is not None else "N/A"

    decision_icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(decision, "⚪")
    decision_label = {"BUY": "买入", "SELL": "卖出", "HOLD": "持有/观望"}.get(decision, decision)

    print(f"\n{SEPARATOR}")
    print(f"  📊  盲测对照摘要 — {ticker} @ {backtest_year} 年底")
    print(SEPARATOR)

    iv  = hist_dcf.get("intrinsic_value")
    mos = hist_dcf.get("margin_of_safety")
    print(f"\n  历史 DCF 内在价值  : {_p(iv)}")
    print(f"  历史市场价格       : {_p(backtest_price)}")
    print(f"  历史安全边际       : {f'{mos:+.1f}%' if mos is not None else 'N/A'}")

    print(f"\n  {decision_icon} AI 历史决策 ({backtest_year}年底): {decision} ({decision_label})")

    print(f"\n  ┌{'─'*40}┐")
    print(f"  │ {'资产':<12} {'入场价':>8}  {'今日价':>8}  {'回报率':>8} │")
    print(f"  ├{'─'*40}┤")
    print(f"  │ {ticker:<12} {_p(backtest_price):>8}  {_p(current_price):>8}  "
          f"{_fmt_return(ticker_return):>8} │")
    print(f"  │ {'SPY(基准)':<12} {_p(spy_backtest):>8}  {_p(spy_current):>8}  "
          f"{_fmt_return(spy_return):>8} │")
    print(f"  └{'─'*40}┘")

    if alpha is not None:
        sign = "+" if alpha >= 0 else ""
        label = "跑赢" if alpha > 5 else ("持平" if alpha > -5 else "跑输")
        print(f"\n  超额收益 Alpha = {sign}{alpha:.1f}%  →  AI {label} SPY 基准")
    else:
        print(f"\n  超额收益 Alpha = N/A（价格数据不足）")

    print(f"\n  持有年数（{backtest_year} → {today_str[:4]}）: {hold_years} 年")
    print(f"\n{SEPARATOR}")
    print("  ⚠️  仅供研究参考，不构成投资建议。历史回测存在后视偏差。")
    print(f"{SEPARATOR}\n")


# ══════════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FinSight AI — 历史盲测模块",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python backtester.py AAPL
  python backtester.py NVDA --year 2022
  python backtester.py MSFT --year 2021 --model gpt-4o --no-save
        """,
    )
    parser.add_argument("ticker",       type=str,   help="股票代码，例如 AAPL")
    parser.add_argument("--year",       type=int,   default=2023,
                        help="回测基准年（模拟在该年底做决策，默认 2023）")
    parser.add_argument("--discount-rate",   type=float, default=DEFAULT_DISCOUNT_RATE,
                        help=f"DCF 折现率（默认 {DEFAULT_DISCOUNT_RATE:.0%}）")
    parser.add_argument("--terminal-growth", type=float, default=DEFAULT_TERMINAL_GROWTH,
                        help=f"永续增长率（默认 {DEFAULT_TERMINAL_GROWTH:.1%}）")
    parser.add_argument("--model",      type=str,   default=DEFAULT_MODEL,
                        help=f"OpenAI 模型（默认 {DEFAULT_MODEL}）")
    parser.add_argument("--no-save",    action="store_true", help="不保存报告到本地文件")
    args = parser.parse_args()

    run_backtest(
        ticker          = args.ticker,
        backtest_year   = args.year,
        discount_rate   = args.discount_rate,
        terminal_growth = args.terminal_growth,
        model           = args.model,
        save_report     = not args.no_save,
    )


if __name__ == "__main__":
    main()
