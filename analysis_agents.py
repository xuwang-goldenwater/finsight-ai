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
import re
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
    "analyst":   900,   # Layer 1 因果分析 + 增速预测
    "broker":    400,   # Layer 2 多空 JSON
    "committee": 1800,  # Layer 3 Dalio 五步法报告（五章节，需要更多空间）
    "chat":      600,   # 投委会实时对话
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

# 基础噪音词：含这些词的结果直接丢弃
_NOISE_KEYWORDS = {
    "advertisement", "sponsored", "cookie", "subscribe now",
    "sign up", "newsletter", "privacy policy", "terms of use",
}

# ── 专业财经来源白名单 ───────────────────────────────────────────
# 附加在 text() 查询末尾，将结果强制限定于顶级机构媒体
_WHITELIST_SUFFIX = (
    "site:reuters.com OR site:bloomberg.com OR site:wsj.com "
    "OR site:cnbc.com OR site:barrons.com OR site:morningstar.com "
    "OR site:ft.com OR site:marketwatch.com"
)

# URL 域名白名单 — 用于标记 open-fallback 结果的可信度
_AUTHORITY_DOMAINS = {
    "reuters.com", "bloomberg.com", "wsj.com", "cnbc.com",
    "barrons.com", "morningstar.com", "ft.com", "marketwatch.com",
    "seekingalpha.com", "finance.yahoo.com", "investing.com",
    "businessinsider.com", "fortune.com", "fool.com",
    "thestreet.com", "benzinga.com", "ap.org", "apnews.com",
}

def _is_authority_url(url: str) -> bool:
    url_lower = url.lower()
    return any(d in url_lower for d in _AUTHORITY_DOMAINS)


def _generate_search_queries(ticker: str, client: OpenAI,
                              model: str = DEFAULT_MODEL) -> list[str]:
    """
    让 LLM 生成 2 条高质量英文搜索词（不含 site: 限定符）。
    site: 限定符由 fetch_live_market_news() 根据搜索层级动态附加。
    失败时返回内置默认词，确保流程不中断。
    """
    system = (
        "You are a financial research assistant. "
        "Output ONLY a JSON array of exactly 2 search query strings, no extra text. "
        "Focus on analyst ratings, price targets, earnings estimates. "
        "Do NOT include any site: operators in the queries."
    )
    user = (
        f"Generate 2 precise English search queries to find the latest "
        f"analyst ratings, price target changes, and earnings estimate revisions for "
        f"{ticker.upper()} stock. Focus on 2025-2026 institutional research. "
        f'Example: ["AAPL analyst rating price target 2025 2026", '
        f'"Apple EPS estimate earnings revision institutional 2026"]'
    )
    try:
        raw = _chat(client, system, user, model=model,
                    temperature=0.3, max_tokens=120)
        cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        queries = json.loads(cleaned)
        if isinstance(queries, list) and len(queries) >= 1:
            return [str(q) for q in queries[:2]]
    except Exception:
        pass
    return [
        f"{ticker.upper()} analyst rating price target 2025 2026",
        f"{ticker.upper()} earnings estimate EPS forecast institutional research 2026",
    ]


def fetch_live_market_news(ticker: str,
                           client: OpenAI,
                           model: str = DEFAULT_MODEL,
                           max_per_query: int = 5) -> str:
    """
    三级降噪新闻抓取策略：

    Tier 1 — DDGS().news()
        专用新闻 API，天然过滤论坛/博客，只返回商业新闻摘要。
        无需 site: 限定符。

    Tier 2 — DDGS().text() + site: 白名单
        严格限定 Reuters / Bloomberg / WSJ / CNBC / Barron's 等顶级媒体。
        用于 Tier 1 结果不足时。

    Tier 3 — DDGS().text() 无限制（冷门股 fallback）
        去掉 site: 限制再搜一次；非权威来源结果打⚠️标记，
        LLM Bouncer 会在认知层进一步过滤。

    每层获得 ≥ 2 条有效结果即停，不再进入下一层。
    """
    if not _HAS_DDG:
        return ""

    queries   = _generate_search_queries(ticker, client, model)
    today_str = datetime.now().strftime("%Y-%m-%d")

    # ── 内部辅助：news() ────────────────────────────────────────
    def _tier1_news(qs: list[str], n: int) -> list[str]:
        seen: set = set()
        out: list[str] = []
        for q in qs:
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.news(q, max_results=n))
                for r in results:
                    url    = (r.get("url") or r.get("href", "")).strip()
                    title  = (r.get("title") or "").strip()
                    body   = (r.get("body") or r.get("excerpt") or "").strip()
                    source = (r.get("source") or "").strip()
                    if url in seen or not (title or body):
                        continue
                    if any(kw in (title + body).lower() for kw in _NOISE_KEYWORDS):
                        continue
                    seen.add(url)
                    body_s = body[:300] + ("…" if len(body) > 300 else "")
                    src_tag = f" · {source}" if source else ""
                    out.append(f"[{title}{src_tag}]\n{body_s}\n({url})")
            except Exception as e:
                print(f"  [DDG.news] '{q}' 失败: {e}")
        return out

    # ── 内部辅助：text() 带可选 site: 后缀 ──────────────────────
    def _tier_text(qs: list[str], n: int, suffix: str = "") -> list[str]:
        seen: set = set()
        out: list[str] = []
        for q in qs:
            full_q = f"{q} {suffix}".strip() if suffix else q
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(full_q, max_results=n))
                for r in results:
                    url   = r.get("href", "").strip()
                    title = (r.get("title") or "").strip()
                    body  = (r.get("body")  or "").strip()
                    if url in seen or not body:
                        continue
                    if any(kw in (title + body).lower() for kw in _NOISE_KEYWORDS):
                        continue
                    seen.add(url)
                    body_s = body[:300] + ("…" if len(body) > 300 else "")
                    # 标记非权威来源，供 LLM Bouncer 参考
                    flag = "" if _is_authority_url(url) else " ⚠️[non-authority-source]"
                    out.append(f"[{title}{flag}]\n{body_s}\n({url})")
            except Exception as e:
                print(f"  [DDG.text] '{full_q}' 失败: {e}")
        return out

    # ── 三级执行 ─────────────────────────────────────────────────
    snippets: list[str] = []
    tier_label = ""

    print(f"  [News] Tier-1: DDGS.news() …")
    snippets = _tier1_news(queries, max_per_query)
    if len(snippets) >= 2:
        tier_label = "DuckDuckGo News API — professional media feed"
        print(f"  [News] ✅ Tier-1 命中 {len(snippets)} 条")
    else:
        print(f"  [News] Tier-1 不足，尝试 Tier-2: text + site whitelist …")
        snippets = _tier_text(queries, max_per_query, suffix=_WHITELIST_SUFFIX)
        if len(snippets) >= 2:
            tier_label = "Web search · restricted to Reuters/Bloomberg/WSJ/CNBC/Barron's/FT/MW"
            print(f"  [News] ✅ Tier-2 命中 {len(snippets)} 条")
        else:
            print(f"  [News] Tier-2 不足，降级 Tier-3: open fallback …")
            snippets = _tier_text(queries, max_per_query + 3, suffix="")
            tier_label = "⚠️ Open web fallback — non-authority sources included; LLM bouncer active"
            print(f"  [News] ⚠️ Tier-3 fallback {len(snippets)} 条（含非权威来源）")

    if not snippets:
        return ""

    header = (
        f"=== Live Market Intelligence for {ticker.upper()} "
        f"(retrieved {today_str}) ===\n"
        f"[Source tier: {tier_label}]\n"
    )
    return header + "\n\n".join(snippets[:8])


# ══════════════════════════════════════════════════════════════════
# Layer 1 — FinancialAnalystAgent
#   输入: 基本面指标 JSON + DCF 结果 JSON
#   输出: 因果剖析 Markdown
# ══════════════════════════════════════════════════════════════════

_ANALYST_SYSTEM_ZH = """你是基本面量化分析师，同时负责为两阶段 DCF 模型预测增速。用紧凑的 Markdown 输出，无废话。

【绝对时序基准 — 严格遵守】
• 当前绝对基准年份：2026 年。
• 2024 年、2025 年的数据与事件均属【已发生历史事实】，必须使用过去时态：
  "2024年财报显示已实现…" / "2025年已完成…"。
  严禁对 2024/2025 使用"将"、"预计"、"计划"等预测性词汇。
• 仅 2026 年下半年及以后，方可使用"将推出"、"预期"等前瞻性表述。
• 若获取到的数据仅截至 2024 年，必须在分析开头注明：
  "⚠️ 基于2024年历史数据回顾（2025–2026年数据待更新）"

四个章节，每章节不超过 150 字：
## 1. 核心指标因果剖析
## 2. 指标间关键因果链
## 3. DCF 模型盲区说明
## 4. 两阶段增速预测
结合公司行业壁垒、护城河强度及近期市场动态，给出两阶段 FCF 增速预测。
**必须**在第 4 章末尾用以下格式输出两行（不得省略，不得修改格式）：
stage1_growth: X.X%
stage2_growth_start: Y.Y%
（stage1_growth = 未来 1-5 年年均增速；stage2_growth_start = 第 6 年起始过渡增速，需介于 stage1_growth 与永续增长率之间）
请使用标准、通顺的简体中文输出，避免出现乱码、字符错位或排版异常。"""

_ANALYST_SYSTEM_EN = """You are a quantitative fundamental analyst responsible for Two-Stage DCF growth prediction. Output concise Markdown, no filler.

[ABSOLUTE TIME ANCHOR — STRICTLY ENFORCED]
• Current absolute reference year: 2026.
• Data and events from 2024 and 2025 are HISTORICAL FACTS. You MUST use past tense:
  "FY2024 results showed…" / "The company completed X in 2025…"
  NEVER use "will", "expected to", "plans to" when referring to 2024 or 2025.
• Forward-looking language ("will launch", "is expected to") is only permitted for H2 2026 and beyond.
• If the available data only covers up to 2024, you MUST state at the top:
  "⚠️ Based on historical data through FY2024 (2025–2026 data pending update)"

Four sections, each ≤150 words:
## 1. Core Metrics — Causal Analysis
## 2. Key Causal Chain Between Metrics
## 3. DCF Model Blind Spots
## 4. Two-Stage Growth Prediction
Based on the company's industry moat, competitive barriers, and recent market dynamics, predict two-stage FCF growth rates.
**You must** end Section 4 with exactly these two lines (no omissions, no format changes):
stage1_growth: X.X%
stage2_growth_start: Y.Y%
(stage1_growth = avg annual FCF growth rate for years 1-5; stage2_growth_start = transition rate starting year 6, between stage1_growth and terminal growth rate)
Use clean, fluent English. Avoid garbled characters or formatting artifacts."""

_GROWTH_RATE_PATTERN = re.compile(
    r"stage(\d)_growth(?:_start)?\s*:\s*([\d.]+)\s*%",
    re.IGNORECASE,
)

def _parse_growth_rates(text: str) -> tuple:
    """
    从 Agent 输出文本中提取 stage1_growth 和 stage2_growth_start。
    返回 (stage1_float_or_None, stage2_float_or_None)。
    """
    stage1, stage2 = None, None
    for m in _GROWTH_RATE_PATTERN.finditer(text):
        stage_num = int(m.group(1))
        rate      = float(m.group(2)) / 100.0
        if stage_num == 1:
            stage1 = rate
        elif stage_num == 2:
            stage2 = rate
    # Sanity clamp: 0% ~ 50%
    if stage1 is not None:
        stage1 = min(max(stage1, 0.0), 0.50)
    if stage2 is not None:
        stage2 = min(max(stage2, 0.0), 0.40)
    return stage1, stage2


def analyze_financial_metrics(
    metrics_json,
    dcf_json,
    ticker: str = "the stock",
    model: str  = DEFAULT_MODEL,
    lang: str   = "zh",
) -> tuple:
    """
    Layer 1：让 LLM 对基本面指标做因果关系深度解读，并预测两阶段增速。

    参数:
        metrics_json : dict / JSON str — 基本面指标（来自 financial_engine）
        dcf_json     : dict / JSON str — DCF 估值结果
        ticker       : 股票代码，用于 Prompt 个性化
        model        : OpenAI 模型名
        lang         : "zh" 或 "en"

    返回:
        (analysis_text: str, stage1_growth: float | None, stage2_growth_start: float | None)
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
    text = _chat(client, system, user_prompt, model=model,
                 max_tokens=_MAX_TOKENS["analyst"])
    stage1, stage2 = _parse_growth_rates(text)
    return text, stage1, stage2


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
你是一位极其挑剔的华尔街顶级对冲基金情报官。你的使命：\
从以下搜索摘要中蒸馏出真正有价值的机构级信号，绝不被噪音迷惑。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【硬性过滤指令 — 不得妥协，不得例外】

✗ 必须屏蔽的噪音来源（看起来多诱人也要丢弃）：
  - 散户情绪与论坛讨论（Reddit、StockTwits、雪球、股吧、东方财富评论区）
  - 无数据支撑的诱导性软文与自媒体个人分析
  - 社交媒体上的个人预测与情绪化表述
  - 无具体机构来源或具名分析师的"小道消息"
  - 标有 ⚠️[non-authority-source] 的结果须提高警觉，仅在无其他选项时才引用

✅ 只允许提取以下三类高价值机构级信号：
  A. 顶级投行/分析师的评级调整与目标价（必须注明机构名称与目标价数字）
  B. 核心财务数据与EPS预期的量化变化（营收/利润率/FCF/EPS具体数字）
  C. 经权威媒体证实的行业政策变动、监管动态或重大宏观事件

若搜索结果全部属于噪音、无有效信号，必须诚实声明，
并从训练知识中补充机构级信号（标注"(模型知识补充)"）。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【时序红线】当前基准年为2026年。2024/2025年的事件属历史事实，必须用过去时态；
仅2026年下半年及之后可用"将"、"预期"等前瞻词汇。

输出严格 JSON，无任何额外文字：
{"bullish":["观点1","观点2","观点3"],"bearish":["观点1","观点2","观点3"],"consensus":"一句话评级","source":"live_search"}\
"""

_BROKER_SYSTEM_EN = """\
You are an exceptionally discerning Wall Street hedge fund intelligence officer. \
Your mission: distill only genuine institutional-grade signals from the search snippets below. \
You are not a summariser — you are a bouncer who rejects noise at the door.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[HARD FILTER DIRECTIVE — NON-NEGOTIABLE, NO EXCEPTIONS]

✗ You MUST discard the following, no matter how compelling they appear:
  - Retail sentiment and forum chatter (Reddit, StockTwits, social media comments)
  - Unsubstantiated opinion pieces and content-farm articles with no data backing
  - Social media predictions and emotionally charged commentary
  - Any "tip" or rumour without a credible named institutional source
  - Results flagged ⚠️[non-authority-source] should be treated with extra scepticism

✅ You are ONLY permitted to extract signals from these three categories:
  A. Rating changes or price target revisions from tier-1 banks/analysts
     (must cite institution name and specific target price)
  B. Quantified changes in core financial estimates: revenue, EPS, FCF, margin guidance
     (must include specific numbers)
  C. Industry policy shifts, regulatory actions, or macro events confirmed by
     authoritative financial media (Reuters, Bloomberg, WSJ, FT, CNBC, Barron's)

If all search results are noise with no valid signals, state that honestly and supplement
with institutional-grade knowledge from your training data (label as "(model knowledge supplement)").
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[TIME ANCHOR] Current reference year is 2026. Events from 2024/2025 are historical facts \
(past tense only). Forward-looking language only permitted for H2 2026 onward.

Output strict JSON only, no extra text:
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
你是 FinSight AI 投资委员会（由 Ray Dalio《原则》方法论驱动）。
你将被研究的公司视为一台运转于宏观经济与债务周期中的"现金流机器"，\
用系统机器思维拆解其每一个驱动齿轮，而非凭感觉做判断。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【时间校准 & 数据缺口强制补全指令 — 最高优先级，绝对执行】

● 今日绝对日期：{today}
● 本次量化财务数据截止日期：{data_date}
● 数据时差：约 {data_lag} 个月

规则一（时态）— 最高执行优先级：
  2024 年、2025 年及 {data_date} 之前发生的一切均属【已发生历史事实】。
  必须用过去时态描述。

  ❌ 绝对禁止的写法（即使数据来自历史记录也不得使用）：
    × "2024 年将推出 AI 功能…"
    × "公司计划于 2025 年完成重组…"
    × "2025 年将实现盈利…"

  ✅ 唯一正确的写法：
    ✓ "公司已于 2024 年推出 AI 功能…"
    ✓ "2025 年完成重组后，公司已进入…"
    ✓ "2025 年实现盈利，全年利润约…"

  仅对 {today} 之后尚未发生的事件，才可使用"预期"、"将"等前瞻词汇。

规则二（禁止敷衍声明）：
  ⛔ 严禁出现"最新数据待更新"、"数据暂不可用"、"待补充"等推卸性表述。
  若量化数据落后，你必须主动利用大模型知识补全时间差：
  - 引用你所知的该公司最新季报要点、营收/利润趋势
  - 引用分析师对 FY2025/FY2026 的 EPS 预估和 Forward P/E
  - 说明 {data_date} 至今的重大战略进展（产品、并购、市场份额）
  并用以下格式标注知识来源：
  **[模型知识补全 · 截至 {today}]**

规则三（前瞻性）：
  仅对 {today} 之后未发生的事件，方可使用"预期"、"将"等前瞻词汇。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

严格按照达利欧【五步流程法】输出完整 Markdown 报告，每步 ≤220 字：

# FinSight 深度投资报告 — {ticker}
**生成时间**: {date}  |  **数据截止**: {data_date}  |  **方法论**: Ray Dalio《原则》五步流程

---

## 🏢 核心业务概述
> 用 2-3 句话说明该公司的核心商业模式、主要收入来源，以及使其区别于竞争对手的核心竞争优势或护城河。不超过 80 字。

---

## Step 1 · 投资目标与期望（Goals）
> 明确本次评估的核心投资假设：这台现金流机器能否在未来 3-5 年持续产生超额回报？\
站在当前宏观与债务周期节点上，期望的风险收益比是什么？

## Step 2 · 核心问题与风险（Problems & Blindspots）
> 直面最大威胁：结合空头观点与 DCF 模型的结构性盲区（单阶段、静态增速假设等），\
列出 2-3 个最可能令投资假设失效的系统性风险或被市场忽视的盲点。

## Step 3 · 指标因果诊断（Diagnose）
> 透视表象，找根因：高 ROE/毛利率的底层驱动是护城河还是周期红利？\
FCF 质量是否可持续？债务杠杆处于宏观周期的什么阶段？

## Step 4 · 估值模型修正设计（Design / Formulation）
> 基于两阶段 DCF 与 Agent 预测增速，理性修正单点估值的偏差。\
给出合理内在价值区间（悲观 / 基准 / 乐观情景），并说明核心假设差异。

## Step 5 · 最终执行决策（Action）
> 明确且可执行的行动方案。\
必须包含：① BUY / HOLD / SELL 决策标签；\
② 建仓/减仓的价格触发区间；③ 两个关键催化剂或止损条件。

**DECISION: BUY 或 HOLD 或 SELL**（单独一行，格式严格）

---
*仅供研究参考，不构成投资建议。*

请使用标准、通顺的简体中文输出，避免出现乱码、字符错位或排版异常。\
"""

_COMMITTEE_SYSTEM_EN = """\
You are the FinSight AI Investment Committee, operating under Ray Dalio's "Principles" methodology.
You treat every company as a cash-flow machine running inside a macro and debt cycle. \
You use Systems/Machine Thinking — decompose every driver, never rely on gut feel.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[TIME CALIBRATION & DATA-GAP BRIDGING — HIGHEST PRIORITY, ALWAYS ENFORCE]

● Today's absolute date: {today}
● Quantitative financial data covers through: {data_date}
● Data lag: approximately {data_lag} months

Rule 1 (Tense) — HIGHEST EXECUTION PRIORITY:
  All data and events prior to {data_date} are HISTORICAL FACTS. Always use past tense.

  ❌ FORBIDDEN (even if sourced from historical data):
    × "In 2024, the company will launch its AI feature…"
    × "The company plans to complete the restructuring in 2025…"
    × "2025 will mark the first profitable year…"

  ✅ ONLY CORRECT FORMS:
    ✓ "The company launched its AI feature in 2024…"
    ✓ "After completing the 2025 restructuring, the company has…"
    ✓ "FY2025 marked the first profitable year, with earnings of…"

  Forward-looking language ("will", "expected to", "is projected to") is ONLY permitted
  for events that have NOT yet occurred as of {today}.

Rule 2 (No Cop-Out Disclaimers):
  ⛔ NEVER write "latest data pending", "data unavailable", or any equivalent dodge.
  If quantitative data is stale, you MUST actively bridge the gap using your model knowledge:
  - Cite the company's most recent quarterly earnings highlights and revenue/profit trends
  - Reference analyst consensus EPS estimates and Forward P/E for FY2025/FY2026
  - Describe material strategic developments since {data_date} (products, M&A, market share)
  Label any knowledge beyond the quantitative data with:
  **[Model Knowledge Supplement · as of {today}]**

Rule 3 (Forward-Looking):
  Forward-looking language ("will", "expected to", "is projected to") is ONLY permitted
  for events that have NOT yet occurred as of {today}.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output a complete Markdown report strictly following Dalio's Five-Step Process (each step ≤220 words):

# FinSight Deep Investment Report — {ticker}
**Generated**: {date}  |  **Data through**: {data_date}  |  **Framework**: Ray Dalio's Principles

---

## 🏢 Business Overview
> In 2-3 sentences, describe the company's core business model, primary revenue streams, and the key competitive advantage or moat that sets it apart from peers. ≤80 words.

---

## Step 1 · Goals
> State the core investment thesis: can this cash-flow machine deliver superior returns over 3-5 years? \
What is the expected risk/reward at the current point in the macro and debt cycle?

## Step 2 · Problems & Blindspots
> Confront the biggest threats: combining the bear-case arguments and the structural limitations of \
the DCF model (static growth, single-stage assumptions), identify 2-3 systemic risks or market \
blindspots that could invalidate the thesis.

## Step 3 · Diagnose
> Cut through the surface metrics to find root causes: is the high ROE/gross margin driven by a \
durable moat or a cyclical tailwind? Is FCF quality sustainable? Where does debt leverage sit in \
the macro cycle?

## Step 4 · Design / Formulation
> Using the Two-Stage DCF and Agent-predicted growth rates, rationally correct single-point \
valuation bias. Provide an intrinsic value range (bear / base / bull scenarios) with key \
assumption differences.

## Step 5 · Action
> A clear, executable action plan. Must include: \
① BUY / HOLD / SELL decision label; \
② price trigger range for entry/exit; \
③ two key catalysts or stop-loss conditions.

**DECISION: BUY or HOLD or SELL** (on its own line, exact format required)

---
*For research and educational purposes only. Not financial advice.*

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
        dcf_json        : DCF 结果 dict/JSON（应含 latest_data_date 字段）
        causal_analysis : Layer 1 输出的因果分析文本
        broker_views    : Layer 2 输出的多空观点 dict
        model           : LLM 模型

    返回:
        完整 Markdown 报告字符串
    """
    client = _get_client()
    ticker_upper = ticker.upper()
    today_str    = datetime.now().strftime("%Y-%m-%d")
    today_dt     = datetime.now()

    # ── 解析 latest_data_date，计算时差（月数）─────────────
    dcf_dict = dcf_json if isinstance(dcf_json, dict) else {}
    if isinstance(dcf_json, str):
        try:
            dcf_dict = json.loads(dcf_json)
        except Exception:
            dcf_dict = {}

    raw_data_date = dcf_dict.get("latest_data_date") or "unknown"

    if raw_data_date != "unknown":
        try:
            from datetime import datetime as _DT
            data_dt  = _DT.strptime(raw_data_date[:10], "%Y-%m-%d")
            lag_days = (today_dt - data_dt).days
            lag_months = max(0, round(lag_days / 30.4))
        except Exception:
            lag_months = 0
    else:
        lag_months = 0

    data_date_label = raw_data_date[:10] if raw_data_date != "unknown" else "unknown"
    data_lag_label  = f"{lag_months}" if lag_months > 0 else "<1"

    # ── 构建系统 Prompt（填充所有占位符）────────────────────
    base_system = _COMMITTEE_SYSTEM_EN if lang == "en" else _COMMITTEE_SYSTEM_ZH
    system = (
        base_system
        .replace("{ticker}",    ticker_upper)
        .replace("{date}",      today_str)
        .replace("{today}",     today_str)
        .replace("{data_date}", data_date_label)
        .replace("{data_lag}",  data_lag_label)
    )

    # ── 构建用户 Prompt：汇总三层数据 ─────────────────────
    bullish_str = "\n".join(f"  - {b}" for b in broker_views.get("bullish", []))
    bearish_str = "\n".join(f"  - {b}" for b in broker_views.get("bearish", []))
    consensus   = broker_views.get("consensus", "N/A")

    slim_m = _slim_metrics(metrics_json)
    slim_d = _slim_dcf(dcf_json)

    gap_note = ""
    if lag_months >= 12:
        if lang == "en":
            gap_note = (
                f"\n[DATA FRESHNESS ALERT] Quantitative data covers through {data_date_label} "
                f"({data_lag_label} months ago). You MUST bridge the gap using model knowledge "
                f"for FY2025/FY2026 earnings, analyst estimates, and strategic developments. "
                f"Label bridged content with [Model Knowledge Supplement · as of {today_str}].\n"
            )
        else:
            gap_note = (
                f"\n[数据新鲜度警告] 量化数据截至 {data_date_label}（约 {data_lag_label} 个月前）。"
                f"你必须用模型知识补全 {data_date_label} 至今的 FY2025/FY2026 财报摘要、"
                f"分析师EPS预估、战略进展，并标注【模型知识补全 · 截至 {today_str}】。\n"
            )

    user_prompt = (
        f"股票:{ticker_upper}\n"
        f"指标:{_compact_json(slim_m)}\n"
        f"DCF:{_compact_json(slim_d)}\n"
        f"因果分析摘要:{causal_analysis[:600]}\n"
        f"多方:{bullish_str}\n"
        f"空方:{bearish_str}\n"
        f"评级:{consensus}"
        f"{gap_note}"
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
    _fe_data: dict    = None,   # 可选：financial_engine fetch_financials() 数据，用于 DCF 重算
) -> tuple:
    """
    完整三层分析流水线入口。
    Layer 1 同时输出两阶段增速预测，用于重算 DCF 后传入 Layer 3。

    参数:
        ticker       : 股票代码，例如 "AAPL"
        metrics_json : financial_engine.calculate_metrics() 的输出（DataFrame 转 JSON 或 dict）
        dcf_json     : financial_engine.calculate_dcf_value() 的输出（dict）
        use_search   : 是否启用 Google 搜索增强券商情报
        model        : OpenAI 模型（默认 gpt-4o）
        save_report  : 是否将报告保存为本地 Markdown 文件
        _fe_data     : (可选) financial_engine.fetch_financials() 数据，传入时重算两阶段 DCF

    返回:
        (final_report: str, stage1_growth: float | None, stage2_growth_start: float | None, updated_dcf: dict)
    """
    ticker_upper = ticker.upper()
    print(f"\n{'═'*60}")
    print(f"  FinSight AI — 启动三层深度分析: {ticker_upper}")
    print(f"{'═'*60}")

    # Layer 1 — 基本面分析 + 两阶段增速预测
    print(f"\n[Layer 1] FinancialAnalystAgent — 基本面因果剖析 + 增速预测…")
    causal_analysis, stage1_growth, stage2_growth_start = analyze_financial_metrics(
        metrics_json, dcf_json, ticker, model, lang=lang
    )
    print(f"  ✓ 因果分析完成 | stage1={stage1_growth} stage2_start={stage2_growth_start}")

    # 若拿到有效增速预测，用 Two-Stage DCF 重算内在价值
    updated_dcf = dcf_json if isinstance(dcf_json, dict) else {}
    if stage1_growth is not None and stage2_growth_start is not None:
        try:
            import financial_engine as _fe
            _dcf_kw = dict(
                stage1_growth       = stage1_growth,
                stage2_growth_start = stage2_growth_start,
            )
            if _fe_data is not None:
                _dcf_kw["_data"] = _fe_data
            updated_dcf = _fe.calculate_dcf_value(ticker, **_dcf_kw)
            print(f"  ✓ Two-Stage DCF 重算完成: 内在价值 ${updated_dcf.get('intrinsic_value', 'N/A'):.2f}")
        except Exception as e:
            print(f"  [警告] Two-Stage DCF 重算失败: {e}，使用原始 DCF")
            updated_dcf = dcf_json if isinstance(dcf_json, dict) else {}
    else:
        print("  [提示] 未能解析增速预测，使用原始单阶段 DCF")

    # Layer 2
    print(f"\n[Layer 2] BrokerIntelAgent — 获取市场多空观点…")
    broker_views = fetch_broker_views(ticker, use_search=use_search, model=model, lang=lang)
    print("  ✓ 多空情报就绪")

    # Layer 3 — 使用更新后的 DCF（Two-Stage）
    print(f"\n[Layer 3] InvestmentCommitteeAgent — 合成最终报告…")
    final_report = run_investment_committee(
        ticker, metrics_json, updated_dcf,
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
    return final_report, stage1_growth, stage2_growth_start, updated_dcf


# ══════════════════════════════════════════════════════════════════
# 投委会实时对话 — answer_investor_question()
#   LLM 扮演达利欧风格投资顾问，结合量化数据与五步法报告实时回答问题
# ══════════════════════════════════════════════════════════════════

_CHAT_SYSTEM_ZH = """\
你是 FinSight AI 达利欧风格投资顾问，正在陪同投资者分析 {ticker}。
你的对话风格：
  • 系统机器思维——把公司当作处于宏观周期中的现金流机器拆解
  • 直接、有观点，用数据支撑每个判断，不说废话
  • 始终区分"已知事实""合理推断""高度不确定"三个层次
  • 如被问到估值，必须给出区间而非单点，并说明关键假设

【时间校准】今日：{today} | 量化数据截至：{data_date}（约 {data_lag} 个月前）
• {data_date} 之前的所有数据和事件属历史事实，必须用过去时态。
• 若投资者问及量化数据截止日期后的情况，主动用模型知识补全，
  并标注"【模型知识 · 截至 {today}】"，绝不用"数据待更新"敷衍。
• 仅对 {today} 之后未发生的事件使用前瞻性表述。

你手上已有以下背景材料（不要在回复中重复罗列，直接引用相关数字）：
--- 量化指标摘要 ---
{metrics_snapshot}
--- DCF 估值摘要 ---
{dcf_snapshot}
--- 五步法深度报告摘要 ---
{report_snippet}

请用简体中文回答，条理清晰，每次回复 ≤300 字。\
避免出现乱码、字符错位或排版异常。\
"""

_CHAT_SYSTEM_EN = """\
You are the FinSight AI investment advisor in the style of Ray Dalio, currently helping an investor \
analyse {ticker}.
Your conversational style:
  • Systems/Machine Thinking — treat the company as a cash-flow machine inside a macro cycle
  • Direct and opinionated; back every claim with data
  • Always distinguish "known facts" / "reasonable inference" / "highly uncertain"
  • When asked about valuation, give a range, never a single point, and state key assumptions

[TIME CALIBRATION] Today: {today} | Quantitative data covers through: {data_date} (~{data_lag} months ago)
• All data and events before {data_date} are historical facts — always past tense.
• If the investor asks about anything after {data_date}, proactively bridge the gap using your
  model knowledge of recent earnings, analyst estimates, and strategic developments.
  Label it: "[Model Knowledge · as of {today}]" — never say "data pending" or "not available".
• Forward-looking language is only permitted for events that have not yet occurred as of {today}.

You have the following background material (do NOT recite it; reference relevant numbers directly):
--- Quantitative Metrics Summary ---
{metrics_snapshot}
--- DCF Valuation Summary ---
{dcf_snapshot}
--- Five-Step Report Excerpt ---
{report_snippet}

Reply in fluent English, ≤300 words per response. Avoid garbled characters or formatting artifacts.\
"""


def answer_investor_question(
    user_query: str,
    ticker: str,
    metrics_json,
    dcf_json,
    agent_report: str  = "",
    chat_history: list = None,   # list of {"role": "user"|"assistant", "content": str}
    model: str         = DEFAULT_MODEL,
    lang: str          = "zh",
) -> str:
    """
    投委会实时问答。

    参数:
        user_query   : 投资者的提问文本
        ticker       : 股票代码
        metrics_json : 基本面指标 dict/JSON
        dcf_json     : DCF 结果 dict/JSON
        agent_report : 已生成的五步法报告（Markdown 字符串，可为空）
        chat_history : 历史对话列表，格式 [{"role": "user", "content": "..."}, ...]
        model        : LLM 模型名
        lang         : "zh" | "en"

    返回:
        LLM 回复字符串
    """
    client = _get_client()

    # 构建上下文摘要（截断防超 token）
    slim_m  = _slim_metrics(metrics_json)
    slim_d  = _slim_dcf(dcf_json)
    metrics_snapshot = _compact_json(slim_m)[:600]
    dcf_snapshot     = _compact_json(slim_d)[:400]
    report_snippet   = agent_report[:800] if agent_report else "（尚未生成报告）"

    # ── 解析时间元数据（与 committee 保持一致）────────────────
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_dt  = datetime.now()
    dcf_dict  = dcf_json if isinstance(dcf_json, dict) else {}
    if isinstance(dcf_json, str):
        try:
            dcf_dict = json.loads(dcf_json)
        except Exception:
            dcf_dict = {}
    raw_data_date   = dcf_dict.get("latest_data_date") or "unknown"
    data_date_label = raw_data_date[:10] if raw_data_date != "unknown" else "unknown"
    if raw_data_date != "unknown":
        try:
            from datetime import datetime as _DT
            lag_months = max(0, round((today_dt - _DT.strptime(raw_data_date[:10], "%Y-%m-%d")).days / 30.4))
        except Exception:
            lag_months = 0
    else:
        lag_months = 0
    data_lag_label = f"{lag_months}" if lag_months > 0 else "<1"

    base_system = _CHAT_SYSTEM_EN if lang == "en" else _CHAT_SYSTEM_ZH
    system = (
        base_system
        .replace("{ticker}",           ticker.upper())
        .replace("{today}",            today_str)
        .replace("{data_date}",        data_date_label)
        .replace("{data_lag}",         data_lag_label)
        .replace("{metrics_snapshot}", metrics_snapshot)
        .replace("{dcf_snapshot}",     dcf_snapshot)
        .replace("{report_snippet}",   report_snippet)
    )

    # 构建完整 messages（含多轮历史）
    messages = [{"role": "system", "content": system.strip()}]
    if chat_history:
        for turn in chat_history[-10:]:   # 最多保留近 10 轮，控制 token
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_query.strip()})

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.5,
            max_tokens=_MAX_TOKENS["chat"],
            messages=messages,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[对话失败] {e}"


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
