"""
analysis_agents.py
==================
FinSight AI — LLM 驱动的三层投资分析 Agent

架构:
    Layer 1 │ FinancialAnalystAgent   — 基本面指标因果剖析
    Layer 2 │ BrokerIntelAgent        — 市场券商多空观点（模拟 / Search API）
    Layer 3 │ InvestmentCommitteeAgent — 多空辩论 + 最终深度报告合成

用法（独立运行）:
    python analysis_agents.py AAPL

用法（作为模块调用）:
    from analysis_agents import run_full_analysis
    report_md = run_full_analysis("AAPL", metrics_json, dcf_json)
"""

import os
import json
import textwrap
import argparse
from datetime import datetime
from typing import Optional

from openai import OpenAI

# ── DuckDuckGo 搜索依赖（不安装时自动降级）────────────────────────
try:
    from duckduckgo_search import DDGS
    _HAS_DDG = True
except ImportError:
    _HAS_DDG = False

# ══════════════════════════════════════════════════════════════════
# 全局配置
# ══════════════════════════════════════════════════════════════════

DEFAULT_MODEL   = "gpt-4o-mini"     # 成本优先；需要更高质量换 "gpt-4o"
DEFAULT_TIMEOUT = 60                # 单次 LLM 调用超时（秒）

# 各层 max_tokens 上限（精简输出，控制费用）
_MAX_TOKENS = {
    "analyst":   800,   # Layer 1 因果分析
    "broker":    400,   # Layer 2 多空 JSON
    "committee": 1200,  # Layer 3 最终报告
}

# 从环境变量读取 API Key（推荐）；也可在此直接赋值（不建议提交到 Git）
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")


# ══════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════

def _get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "请设置环境变量 OPENAI_API_KEY，例如:\n"
            "  export OPENAI_API_KEY='sk-...'\n"
            "或在 analysis_agents.py 顶部直接赋值 OPENAI_API_KEY。"
        )
    return OpenAI(api_key=OPENAI_API_KEY, timeout=DEFAULT_TIMEOUT)


def _chat(
    client: OpenAI,
    system: str,
    user: str,
    model: str      = DEFAULT_MODEL,
    temperature: float = 0.4,
    max_tokens: int = 1500,
) -> str:
    """封装单次 ChatCompletion 调用，统一异常处理。"""
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system.strip()},
                {"role": "user",   "content": user.strip()},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM 调用失败] {e}"


def _slim_metrics(metrics_json) -> dict:
    """
    精简基本面数据：只保留最新一年 + 5 年均值，去掉冗余历史列。
    大幅减少输入 Token 数。
    """
    if isinstance(metrics_json, str):
        try:
            metrics_json = json.loads(metrics_json)
        except Exception:
            return metrics_json

    if not isinstance(metrics_json, dict) or not metrics_json:
        return metrics_json

    years      = sorted(metrics_json.keys())
    latest_key = years[-1]
    latest     = metrics_json[latest_key]

    # 计算 5 年均值（跳过 None）
    averages = {}
    for metric in latest:
        vals = [metrics_json[y].get(metric) for y in years if metrics_json[y].get(metric) is not None]
        averages[metric] = round(sum(vals) / len(vals), 2) if vals else None

    return {
        "latest_year": latest_key,
        "latest":      {k: (round(v, 2) if isinstance(v, float) else v) for k, v in latest.items()},
        "5yr_avg":     averages,
    }


def _slim_dcf(dcf_json) -> dict:
    """
    精简 DCF 结果：只保留结论层字段，去掉逐年预测列表（节省 ~200 Token）。
    """
    if isinstance(dcf_json, str):
        try:
            dcf_json = json.loads(dcf_json)
        except Exception:
            return dcf_json

    keys = [
        "ticker", "intrinsic_value", "current_price", "margin_of_safety",
        "fcf_growth_rate", "base_fcf", "discount_rate", "terminal_growth",
        "total_pv", "cash", "total_debt", "shares", "error",
    ]
    return {k: dcf_json.get(k) for k in keys if k in dcf_json}


def _compact_json(data) -> str:
    """输出紧凑（无缩进）JSON，进一步减少 Token。"""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return data
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


# ══════════════════════════════════════════════════════════════════
# DDG 实时新闻搜索（BrokerIntelAgent 数据源）
# ══════════════════════════════════════════════════════════════════

# 噪音过滤关键词：含这些词的结果直接跳过
_NOISE_KEYWORDS = {
    "advertisement", "sponsored", "cookie", "subscribe now",
    "sign up", "newsletter", "privacy policy", "terms of use",
}

def _generate_search_queries(ticker: str, client: OpenAI,
                              model: str = DEFAULT_MODEL) -> list[str]:
    """
    让 LLM 为给定股票生成 2 条高质量英文搜索词。
    失败时返回内置默认词，确保流程不中断。
    """
    system = ("You are a financial research assistant. "
              "Output ONLY a JSON array of exactly 2 search query strings, no extra text.")
    user   = (f"Generate 2 precise English search queries to find the latest "
              f"analyst opinions, price targets, and bull/bear thesis for {ticker.upper()} stock. "
              f"Focus on 2025-2026 market dynamics. Example format: "
              f'["AAPL stock analyst rating target price 2025", '
              f'"Apple bull bear case risks opportunities 2025"]')
    try:
        raw = _chat(client, system, user, model=model,
                    temperature=0.3, max_tokens=120)
        cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        queries = json.loads(cleaned)
        if isinstance(queries, list) and len(queries) >= 1:
            return [str(q) for q in queries[:2]]
    except Exception:
        pass
    # 默认回退
    return [
        f"{ticker.upper()} stock analyst rating target price 2025 2026",
        f"{ticker.upper()} bull bear case risks investment thesis 2025",
    ]


def fetch_live_market_news(ticker: str,
                           client: OpenAI,
                           model: str = DEFAULT_MODEL,
                           max_per_query: int = 5) -> str:
    """
    用 DuckDuckGo 实时搜索并返回清洗后的新闻摘要文本。

    流程：
        1. LLM 生成 2 条专业搜索词
        2. DDGS().text() 各取前 max_per_query 条
        3. 去重 + 过滤噪音 + 格式化为 Prompt 友好的纯文本

    返回：
        str — 可直接拼入 Prompt 的新闻摘要；失败时返回空字符串
    """
    if not _HAS_DDG:
        return ""

    queries = _generate_search_queries(ticker, client, model)
    seen_urls: set = set()
    snippets: list[str] = []

    for query in queries:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_per_query))
            for r in results:
                url   = r.get("href", "")
                title = (r.get("title") or "").strip()
                body  = (r.get("body")  or "").strip()

                # 去重
                if url in seen_urls or not body:
                    continue
                # 噪音过滤（标题或摘要含噪音词则跳过）
                combined_lower = (title + " " + body).lower()
                if any(kw in combined_lower for kw in _NOISE_KEYWORDS):
                    continue

                seen_urls.add(url)
                # 截断单条摘要，防止超长
                body_short = body[:300] + ("…" if len(body) > 300 else "")
                snippets.append(f"[{title}]\n{body_short}\n({url})")
        except Exception as e:
            print(f"  [DDG] 查询 '{query}' 失败: {e}")
            continue

    if not snippets:
        return ""

    header = (f"=== Live Market Intelligence for {ticker.upper()} "
              f"(retrieved {datetime.now().strftime('%Y-%m-%d')}) ===\n")
    return header + "\n\n".join(snippets[:8])   # 最多保留 8 条，控制 Token


# ══════════════════════════════════════════════════════════════════
# Layer 1 — FinancialAnalystAgent
#   输入: 基本面指标 JSON + DCF 结果 JSON
#   输出: 因果剖析 Markdown
# ══════════════════════════════════════════════════════════════════

_ANALYST_SYSTEM_ZH = """你是基本面量化分析师。用紧凑的 Markdown 输出，无废话。
三个章节，每章节不超过 150 字：
## 1. 核心指标因果剖析
## 2. 指标间关键因果链
## 3. DCF 模型盲区说明
请使用标准、通顺的简体中文输出，避免出现乱码、字符错位或排版异常。"""

_ANALYST_SYSTEM_EN = """You are a quantitative fundamental analyst. Output concise Markdown, no filler.
Three sections, each ≤150 words:
## 1. Core Metrics — Causal Analysis
## 2. Key Causal Chain Between Metrics
## 3. DCF Model Blind Spots
Use clean, fluent English. Avoid garbled characters or formatting artifacts."""

def analyze_financial_metrics(
    metrics_json,
    dcf_json,
    ticker: str = "the stock",
    model: str  = DEFAULT_MODEL,
    lang: str   = "zh",
) -> str:
    """
    Layer 1：让 LLM 对基本面指标做因果关系深度解读。

    参数:
        metrics_json : dict / JSON str — 基本面指标（来自 financial_engine）
        dcf_json     : dict / JSON str — DCF 估值结果
        ticker       : 股票代码，用于 Prompt 个性化
        model        : OpenAI 模型名

    返回:
        因果分析 Markdown 字符串
    """
    client  = _get_client()
    system  = _ANALYST_SYSTEM_EN if lang == "en" else _ANALYST_SYSTEM_ZH
    slim_m  = _slim_metrics(metrics_json)
    slim_d  = _slim_dcf(dcf_json)

    user_prompt = (
        f"{ticker.upper()} financial data:\n"
        f"Metrics:{_compact_json(slim_m)}\n"
        f"DCF:{_compact_json(slim_d)}"
    )
    return _chat(client, system, user_prompt, model=model,
                 max_tokens=_MAX_TOKENS["analyst"])


# ══════════════════════════════════════════════════════════════════
# Layer 2 — BrokerIntelAgent
#   获取市场券商多空观点（支持真实搜索 / 模拟数据两种模式）
# ══════════════════════════════════════════════════════════════════

# 模拟数据库：覆盖常见股票，其余由 LLM 生成
_SIMULATED_BROKER_VIEWS = {
    "AAPL": {
        "bullish": [
            "AI 驱动 iPhone 超级换机周期：Apple Intelligence 功能推动 16/17 系列升级需求。",
            "服务业务（App Store、Apple TV+、iCloud）毛利率超 70%，占营收比持续提升至 25%+。",
            "生态系统护城河极深：硬件-软件-服务飞轮效应，用户切换成本极高。",
            "印度市场高速扩张，对冲大中华区风险，全球制造多元化布局加速。",
        ],
        "bearish": [
            "硬件创新瓶颈：iPhone 功能同质化严重，消费者换机周期拉长至 4-5 年。",
            "反垄断监管压力：DOJ / EU 对 App Store 抽成模式持续施压，服务收入存在监管风险。",
            "大中华区市场份额持续下滑：华为 Mate 系列回归蚕食高端市场，2024 年中国营收下降 8%。",
            "估值偏高：当前 P/E 约 28-30x，溢价难以单纯用增速支撑，需要 AI 变现落地验证。",
        ],
        "consensus": "超配/买入（约 65% 分析师评级 Buy，平均目标价较当前溢价 8-12%）",
    },
    "MSFT": {
        "bullish": [
            "Azure 云增速重新加速至 29%+，Copilot AI 商业化开始贡献营收。",
            "企业软件生态（Office 365、Teams、LinkedIn）提供稳定高毛利经常性收入。",
            "OpenAI 深度绑定，AI 基础设施领域卡位优势明显。",
        ],
        "bearish": [
            "云计算市场竞争白热化，AWS 和 Google Cloud 持续抢份额。",
            "Activision 收购整合风险：游戏业务贡献低于预期。",
            "估值已 price in 大量 AI 增长预期，若兑现不及预期将面临估值收缩。",
        ],
        "consensus": "买入（约 85% 分析师评级 Buy，市场高度一致看多）",
    },
    "GOOGL": {
        "bullish": [
            "搜索广告护城河依然稳固，AI Overview 提升搜索货币化效率。",
            "Google Cloud 增速加速至 28%+，进入规模化盈利阶段。",
            "Waymo 自动驾驶商业化领先，潜在期权价值尚未计入估值。",
        ],
        "bearish": [
            "AI 搜索替代威胁：ChatGPT、Perplexity 等 AI 原生搜索分流查询量。",
            "反垄断诉讼：DOJ 搜索垄断案判决可能强制拆分或限制 Chrome/Android 默认协议。",
            "广告收入集中度高，宏观经济下行时广告预算首当其冲。",
        ],
        "consensus": "买入（约 80% 分析师评级 Buy）",
    },
    "AMZN": {
        "bullish": [
            "AWS 增速回升至 17%+，AI 基础设施需求推动云业务超级周期。",
            "广告业务高速增长（年增 20%+），成为第三大收入来源且利润率极高。",
            "零售业务持续降本增效，北美电商利润率改善显著。",
        ],
        "bearish": [
            "资本开支大幅增加（AI 数据中心投资），短期 FCF 承压。",
            "国际业务仍亏损，全球扩张成本高企。",
            "监管风险：FTC 反垄断调查持续，第三方卖家政策面临审查。",
        ],
        "consensus": "强力买入（约 90% 分析师评级 Buy）",
    },
    "NVDA": {
        "bullish": [
            "AI 训练芯片市场占有率超 80%，Blackwell 架构需求严重供不应求。",
            "CUDA 软件生态护城河是真正的竞争壁垒，非单纯硬件优势。",
            "数据中心营收占比超 80%，摆脱游戏业务周期性束缚。",
        ],
        "bearish": [
            "估值极度透支增长预期，Forward P/E 约 35-40x，任何业绩 miss 将引发剧烈调整。",
            "AMD MI300 / 自研芯片（Google TPU、Amazon Trainium）潜在替代压力。",
            "出口管制风险：美国对华芯片限制持续收紧，中国市场收入受限。",
        ],
        "consensus": "买入（约 85% 分析师评级 Buy，但目标价分歧较大）",
    },
    "BRK-B": {
        "bullish": [
            "持有大量现金（约 1800 亿美元），熊市中具备极强的逆势收购能力。",
            "保险浮存金提供低成本杠杆，保险业务盈利能力持续改善。",
            "投资组合质量高，AAPL 等核心仓位提供稳定股息与增值。",
        ],
        "bearish": [
            "巴菲特年龄风险：接班人能力尚未得到市场充分验证。",
            "规模诅咒：体量过大导致超额收益越来越难以实现。",
            "重仓传统行业，科技转型敞口不足，在 AI 浪潮中相对滞后。",
        ],
        "consensus": "持有/买入（分析师评级偏保守，认可安全边际但成长性有限）",
    },
}

_BROKER_SYSTEM_ZH = """\
你是顶级卖方研究员。根据下方【实时新闻情报】提炼多空观点。
规则：
- 必须从新闻内容中归纳，不得凭空捏造
- 若新闻不足，可补充行业常识，但需标注"(背景知识)"
- 输出严格 JSON，无任何额外文字：
{"bullish":["观点1","观点2","观点3"],"bearish":["观点1","观点2","观点3"],"consensus":"一句话评级","source":"live_search"}\
"""

_BROKER_SYSTEM_EN = """\
You are a top sell-side analyst. Extract bull/bear thesis from the LIVE NEWS below.
Rules:
- Ground each point in the news; do NOT fabricate
- If news is insufficient, supplement with industry knowledge and mark as "(background)"
- Output strict JSON only, no extra text:
{"bullish":["point1","point2","point3"],"bearish":["point1","point2","point3"],"consensus":"one-sentence rating","source":"live_search"}\
"""

# 无新闻时的降级 Prompt（不要求从新闻归纳）
_BROKER_SYSTEM_FALLBACK_ZH = (
    "输出严格 JSON，无任何额外文字：\n"
    '{"bullish":["观点1","观点2","观点3"],'
    '"bearish":["观点1","观点2","观点3"],'
    '"consensus":"一句话评级","source":"llm_knowledge"}'
)
_BROKER_SYSTEM_FALLBACK_EN = (
    "Output strict JSON only, no extra text:\n"
    '{"bullish":["point1","point2","point3"],'
    '"bearish":["point1","point2","point3"],'
    '"consensus":"one-sentence rating","source":"llm_knowledge"}'
)


def _parse_broker_json(raw: str) -> dict | None:
    """清理并解析 LLM 返回的 broker JSON，失败返回 None。"""
    try:
        cleaned = raw.strip()
        # 剥离 markdown code fence
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:] if lines[0].startswith("```") else lines)
        cleaned = cleaned.rstrip("`").strip()
        result = json.loads(cleaned)
        if isinstance(result.get("bullish"), list) and isinstance(result.get("bearish"), list):
            return result
    except Exception:
        pass
    return None


def fetch_broker_views(
    ticker: str,
    use_search: bool = True,       # 默认开启 DDG 搜索
    model: str       = DEFAULT_MODEL,
    lang: str        = "zh",
) -> dict:
    """
    Layer 2 — BrokerIntelAgent（DDG 实时 RAG 版）

    优先级链：
        1. DDG 实时搜索 → LLM 从新闻提炼多空
        2. 本地 Mock 数据库（覆盖 AAPL/MSFT/NVDA 等主流股）
        3. LLM 凭训练知识推断（兜底）

    返回 dict，含 bullish / bearish / consensus / source 四个 key。
    source 值：live_search | mock_db | llm_knowledge
    """
    ticker_upper = ticker.upper()
    client = _get_client()

    # ══ 阶段 1：DDG 实时搜索 ══════════════════════════════════════
    if use_search and _HAS_DDG:
        print(f"  [BrokerIntel] 🌐 DDG 实时搜索 {ticker_upper}…")
        try:
            news_text = fetch_live_market_news(ticker_upper, client, model)
        except Exception as e:
            print(f"  [BrokerIntel] DDG 搜索异常: {e}，降级处理")
            news_text = ""

        if news_text:
            broker_system = _BROKER_SYSTEM_EN if lang == "en" else _BROKER_SYSTEM_ZH
            output_lang   = "English" if lang == "en" else "中文"
            user_prompt = (
                f"Stock: {ticker_upper}. "
                f"Extract 3 bullish and 3 bearish points from the news below. "
                f"Output language: {output_lang}.\n\n"
                f"{news_text}"
            )
            raw = _chat(client, broker_system, user_prompt, model=model,
                        temperature=0.3, max_tokens=_MAX_TOKENS["broker"] + 100)
            result = _parse_broker_json(raw)
            if result:
                print(f"  [BrokerIntel] ✅ 实时新闻提炼成功 (source: live_search)")
                result["source"] = "live_search"
                return result
            print(f"  [BrokerIntel] ⚠️ JSON 解析失败，进入降级")
        else:
            print(f"  [BrokerIntel] ⚠️ DDG 未返回有效结果，进入降级")

    elif use_search and not _HAS_DDG:
        print(f"  [BrokerIntel] ⚠️ duckduckgo_search 未安装，跳过实时搜索")
        print(f"       安装命令: pip install duckduckgo-search")

    # ══ 阶段 2：本地 Mock 数据库 ══════════════════════════════════
    if ticker_upper in _SIMULATED_BROKER_VIEWS:
        local_data = dict(_SIMULATED_BROKER_VIEWS[ticker_upper])
        local_data["source"] = "mock_db"

        if lang != "en":
            print(f"  [BrokerIntel] 📦 使用本地 Mock 数据库 ({ticker_upper})")
            return local_data

        # 英文模式：LLM 将中文 Mock 翻译为英文
        print(f"  [BrokerIntel] 📦 Mock 数据 → LLM 英文转述")
        zh_bull = "\n".join(local_data.get("bullish", []))
        zh_bear = "\n".join(local_data.get("bearish", []))
        raw = _chat(
            client, _BROKER_SYSTEM_FALLBACK_EN,
            (f"Translate into concise English for {ticker_upper}. 3 points each.\n\n"
             f"Bull:\n{zh_bull}\n\nBear:\n{zh_bear}\n\n"
             f"Consensus: {local_data.get('consensus','')}"),
            model=model, temperature=0.3, max_tokens=_MAX_TOKENS["broker"],
        )
        result = _parse_broker_json(raw)
        if result:
            result["source"] = "mock_db"
            return result
        local_data["source"] = "mock_db"
        return local_data   # 翻译失败时返回中文原文

    # ══ 阶段 3：LLM 训练知识兜底 ══════════════════════════════════
    print(f"  [BrokerIntel] 🤖 LLM 训练知识推断 {ticker_upper}…")
    fallback_system = (_BROKER_SYSTEM_FALLBACK_EN if lang == "en"
                       else _BROKER_SYSTEM_FALLBACK_ZH)
    output_lang = "English" if lang == "en" else "中文"
    user_prompt = (
        f"{ticker_upper}: 3 bull and 3 bear points, 2024-2025 dynamics. "
        f"Output: {output_lang}. Mark uncertain points with (estimated)."
    )
    raw = _chat(client, fallback_system, user_prompt, model=model,
                temperature=0.5, max_tokens=_MAX_TOKENS["broker"])
    result = _parse_broker_json(raw)
    if result:
        result["source"] = "llm_knowledge"
        return result

    # 最终兜底：返回结构化空值，绝不崩溃
    return {
        "bullish":   ["暂时无法联网，改用模型常识分析（数据获取失败）"] if lang != "en"
                     else ["Unable to retrieve live data; using model knowledge (fallback)"],
        "bearish":   ["暂时无法联网，改用模型常识分析（数据获取失败）"] if lang != "en"
                     else ["Unable to retrieve live data; using model knowledge (fallback)"],
        "consensus": "数据获取失败，请稍后重试" if lang != "en" else "Data unavailable, please retry",
        "source":    "error",
    }


# ══════════════════════════════════════════════════════════════════
# Layer 3 — InvestmentCommitteeAgent
#   输入: 因果分析 + 券商多空 + 量化指标
#   输出: FinSight 深度投资报告（Markdown）
# ══════════════════════════════════════════════════════════════════

_COMMITTEE_SYSTEM_ZH = """\
你是 FinSight AI 投资委员会。风格：巴菲特式价值投资，直接有观点，无废话。
严格按以下 Markdown 结构输出（每章节 ≤200 字）：

# FinSight 深度投资报告 — {ticker}
**生成时间**: {date}

## 一、核心量化快照
## 二、指标因果剖析
## 三、多空风向标
## 四、DCF 修正说明
## 五、综合投资建议（必须给出 买入/持有/回避 + 2个关键催化剂）

---
*仅供参考，不构成投资建议。*

请使用标准、通顺的简体中文输出，避免出现乱码、字符错位或排版异常。\
"""

_COMMITTEE_SYSTEM_EN = """\
You are the FinSight AI Investment Committee. Style: Buffett-style value investing — direct, opinionated, no filler.
Output strictly in English using the following Markdown structure (each section ≤200 words):

# FinSight Deep Investment Report — {ticker}
**Generated**: {date}

## I. Core Quantitative Snapshot
## II. Metrics Causal Analysis
## III. Bull vs Bear Signals
## IV. DCF Adjustment Notes
## V. Investment Recommendation (must state BUY / HOLD / AVOID + 2 key catalysts)

---
*For research purposes only. Not financial advice.*

Use clean, fluent English only. Avoid garbled characters or formatting artifacts.\
"""

def run_investment_committee(
    ticker: str,
    metrics_json,
    dcf_json,
    causal_analysis: str,
    broker_views: dict,
    model: str = DEFAULT_MODEL,
    lang: str  = "zh",
) -> str:
    """
    Layer 3：投委会 Agent，输出最终 FinSight 深度投资报告。

    参数:
        ticker          : 股票代码
        metrics_json    : 基本面指标 dict/JSON
        dcf_json        : DCF 结果 dict/JSON
        causal_analysis : Layer 1 输出的因果分析文本
        broker_views    : Layer 2 输出的多空观点 dict
        model           : LLM 模型

    返回:
        完整 Markdown 报告字符串
    """
    client = _get_client()
    ticker_upper = ticker.upper()
    today = datetime.now().strftime("%Y-%m-%d")

    # ── 构建系统 Prompt（填充 ticker 和日期）──────────────
    base_system = _COMMITTEE_SYSTEM_EN if lang == "en" else _COMMITTEE_SYSTEM_ZH
    system = base_system.replace("{ticker}", ticker_upper).replace("{date}", today)

    # ── 构建用户 Prompt：汇总三层数据 ─────────────────────
    bullish_str = "\n".join(f"  - {b}" for b in broker_views.get("bullish", []))
    bearish_str = "\n".join(f"  - {b}" for b in broker_views.get("bearish", []))
    consensus   = broker_views.get("consensus", "N/A")

    slim_m = _slim_metrics(metrics_json)
    slim_d = _slim_dcf(dcf_json)

    user_prompt = (
        f"股票:{ticker_upper}\n"
        f"指标:{_compact_json(slim_m)}\n"
        f"DCF:{_compact_json(slim_d)}\n"
        f"因果分析摘要:{causal_analysis[:600]}\n"   # 截断防超长
        f"多方:{bullish_str}\n"
        f"空方:{bearish_str}\n"
        f"评级:{consensus}"
    )
    return _chat(client, system, user_prompt, model=model,
                 temperature=0.6, max_tokens=_MAX_TOKENS["committee"])


# ══════════════════════════════════════════════════════════════════
# 公开 API：一键运行完整三层分析
# ══════════════════════════════════════════════════════════════════

def run_full_analysis(
    ticker: str,
    metrics_json,
    dcf_json,
    use_search: bool  = False,
    model: str        = DEFAULT_MODEL,
    save_report: bool = True,
    lang: str         = "zh",
) -> str:
    """
    完整三层分析流水线入口。

    参数:
        ticker       : 股票代码，例如 "AAPL"
        metrics_json : financial_engine.calculate_metrics() 的输出（DataFrame 转 JSON 或 dict）
        dcf_json     : financial_engine.calculate_dcf_value() 的输出（dict）
        use_search   : 是否启用 Google 搜索增强券商情报
        model        : OpenAI 模型（默认 gpt-4o）
        save_report  : 是否将报告保存为本地 Markdown 文件

    返回:
        最终报告的 Markdown 字符串
    """
    ticker_upper = ticker.upper()
    print(f"\n{'═'*60}")
    print(f"  FinSight AI — 启动三层深度分析: {ticker_upper}")
    print(f"{'═'*60}")

    # Layer 1
    print(f"\n[Layer 1] FinancialAnalystAgent — 基本面因果剖析…")
    causal_analysis = analyze_financial_metrics(metrics_json, dcf_json, ticker, model, lang=lang)
    print("  ✓ 因果分析完成")

    # Layer 2
    print(f"\n[Layer 2] BrokerIntelAgent — 获取市场多空观点…")
    broker_views = fetch_broker_views(ticker, use_search=use_search, model=model, lang=lang)
    print("  ✓ 多空情报就绪")

    # Layer 3
    print(f"\n[Layer 3] InvestmentCommitteeAgent — 合成最终报告…")
    final_report = run_investment_committee(
        ticker, metrics_json, dcf_json,
        causal_analysis, broker_views, model, lang=lang
    )
    print("  ✓ 深度报告生成完毕")

    # 可选：保存报告到本地
    if save_report:
        filename = f"finsight_report_{ticker_upper}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(final_report)
            print(f"\n  📄 报告已保存: {filename}")
        except Exception as e:
            print(f"\n  [警告] 保存报告失败: {e}")

    print(f"\n{'═'*60}\n")
    return final_report


# ══════════════════════════════════════════════════════════════════
# CLI 入口 — 与 financial_engine.py 联动
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FinSight AI — LLM 深度投资分析（需先运行 financial_engine.py）"
    )
    parser.add_argument("ticker", type=str, help="股票代码，例如 AAPL")
    parser.add_argument("--model",      default=DEFAULT_MODEL, help="OpenAI 模型（默认 gpt-4o）")
    parser.add_argument("--use-search", action="store_true",   help="启用 Google 搜索增强")
    parser.add_argument("--no-save",    action="store_true",   help="不保存报告到本地文件")
    args = parser.parse_args()
    ticker = args.ticker.upper()

    # ── 公司确认 ──────────────────────────────────────────────────
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        company_name, last_price = None, None
        try:
            info = tk.info or {}
            company_name = info.get("longName") or info.get("shortName")
        except Exception:
            pass
        try:
            fi = getattr(tk, "fast_info", None)
            last_price = getattr(fi, "last_price", None) if fi else None
        except Exception:
            pass

        if not last_price and not company_name:
            print(f"\n  ⚠️  找不到 [{ticker}] 的有效数据，请确认代码是否正确。")
            print(f"      常见示例：AppLovin → APP  |  苹果 → AAPL  |  英伟达 → NVDA")
            return

        display    = company_name or ticker
        price_str  = f"${last_price:.2f}" if last_price else "价格暂不可用"
        print(f"\n  找到公司: 【{display}】({ticker})  当前价格: {price_str}")
        confirm = input("  确认是这家公司吗？按 Enter 继续，输入 n 退出: ").strip().lower()
        if confirm == "n":
            print("  已取消，请重新输入正确的股票代码。")
            return
    except Exception as e:
        print(f"  ⚠️  代码验证失败: {e}，将直接继续…")

    # 从 financial_engine 拉取数据
    try:
        from financial_engine import analyze
    except ImportError:
        print("[错误] 请确保 financial_engine.py 与本文件在同一目录。")
        return

    print(f"\n⏳  正在获取 {ticker} 财务数据（来自 financial_engine）…")
    try:
        metrics_df, dcf_result, _ = analyze(ticker)
    except Exception as e:
        print(f"[错误] financial_engine 运行失败: {e}")
        return

    # DataFrame → JSON（orient="index" 保留年份索引）
    if hasattr(metrics_df, "to_json"):
        metrics_json = json.loads(metrics_df.to_json(orient="index"))
    else:
        metrics_json = metrics_df

    # 过滤 dcf_result 中不可序列化的字段
    dcf_json = {
        k: v for k, v in dcf_result.items()
        if k != "ticker_obj" and isinstance(v, (str, int, float, list, dict, type(None)))
    }

    # 运行三层分析
    report = run_full_analysis(
        ticker,
        metrics_json,
        dcf_json,
        use_search = args.use_search,
        model      = args.model,
        save_report= not args.no_save,
    )

    print(report)


if __name__ == "__main__":
    main()
