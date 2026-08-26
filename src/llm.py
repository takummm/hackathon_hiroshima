"""LLM連携（ステップ④原因分析・⑤簡易RAG施策提案）。

- ④ explain_risk_factors(): リスクスコア＋feature importanceを自然文で説明
- ⑤ suggest_interventions(): 企業制度テキスト（簡易RAG）を踏まえた定着施策の提案

開発中はHaiku 4.5（低コスト・高速）、デモ本番はSonnet 5（品質優先）を使い分ける。
API呼び出しの不安定さに備え、プロンプトは短く固定し、Streamlit側でst.cache_dataによる
レスポンスキャッシュを行う想定。
"""

from __future__ import annotations

import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

POLICIES_PATH = Path(__file__).resolve().parent.parent / "data" / "company_policies.txt"

MODEL_DEV = "claude-haiku-4-5-20251001"
MODEL_PROD = "claude-sonnet-5"

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY が .env に設定されていません。")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def load_company_policies() -> str:
    return POLICIES_PATH.read_text(encoding="utf-8")


def friendly_error_message(e: Exception) -> str:
    """Claude API呼び出し失敗時に、ユーザー向けの分かりやすいメッセージに変換する。"""
    if isinstance(e, anthropic.APITimeoutError):
        return "AIサーバーへの応答がタイムアウトしました。少し時間をおいてもう一度お試しください。"
    if isinstance(e, anthropic.RateLimitError):
        return "現在AIへのリクエストが混み合っています。少し時間をおいてもう一度お試しください。"
    if isinstance(e, anthropic.AuthenticationError):
        return "AI連携の認証に失敗しました。管理者にAPIキーの設定を確認してもらってください。"
    if isinstance(e, anthropic.APIConnectionError):
        return "AIサーバーに接続できませんでした。ネットワーク状況を確認し、もう一度お試しください。"
    if isinstance(e, anthropic.APIStatusError):
        return f"AIサービス側でエラーが発生しました（コード: {e.status_code}）。時間をおいて再度お試しください。"
    if isinstance(e, anthropic.APIError):
        return "AI呼び出し中に問題が発生しました。時間をおいて再度お試しください。"
    return "予期しないエラーが発生しました。時間をおいて再度お試しください。"


def get_risk_tier(risk_pct: float) -> str:
    """デモ用の3段階リスク区分。高:70%以上／中:30〜50%／低:10%以下（それ以外は中間扱い）。"""
    if risk_pct >= 70:
        return "high"
    if risk_pct <= 10:
        return "low"
    return "mid"


_EXPLANATION_TONE = {
    "high": (
        "このリスクは非常に高い水準です。深刻な状況であることが伝わるよう、"
        "「重大な懸念があります」「早急な対応が必要です」のように緊迫感のある表現を用いてください。"
    ),
    "mid": (
        "このリスクは中程度の水準です。過度に深刻にも楽観的にもせず、"
        "「中期的に注意が必要です」のような落ち着いたトーンで記述してください。"
    ),
    "low": (
        "このリスクは低い水準です。「現時点で特に大きな懸念事項はありません」という趣旨を明確に伝え、"
        "安心感のある文章にしてください。存在しないリスクを無理に作り出さないでください。"
    ),
}

_INTERVENTION_TONE = {
    "high": (
        "緊急対応レベルの施策として提案してください。「直ちに」「早急に」「今週中に」のように、"
        "優先度が高く即座の行動を促す言葉を使ってください。"
    ),
    "mid": (
        "中期的な改善施策として提案してください。「今後数ヶ月かけて」「計画的に」のように、"
        "落ち着いたペースでの取り組みを促す言葉を使ってください。"
    ),
    "low": (
        "施策というよりも、現状維持・継続観察を中心とした内容にしてください。"
        "「特別な対応は不要ですが、〜を継続することを推奨します」のように、"
        "問題があるかのような過度な提案は避けてください。"
    ),
}


def explain_risk_factors(
    employee_summary: str,
    risk_pct: float,
    factors_text: str,
    model: str = MODEL_DEV,
) -> str:
    """④ リスクスコア＋feature importance上位を自然文で説明する。

    金銭・通勤面だけでなく満足度面（ワークライフバランス・キャリア成長等）も
    バランスよく言及するよう指示する。
    """
    tone = _EXPLANATION_TONE[get_risk_tier(risk_pct)]

    prompt = f"""あなたは人事データ分析の専門家です。以下の従業員について、離職リスクの
判定理由を、日本語で3〜4文の自然な文章で説明してください。

# 従業員プロフィール
{employee_summary}

# 離職リスクスコア
{risk_pct:.1f}%

# 主要リスク要因（feature importance上位、値が大きいほどモデルへの影響が大きい）
{factors_text}

# 指示
- 数値の羅列ではなく、要因同士のつながりを踏まえた自然な説明にすること
- 金銭・通勤などの条件面だけでなく、ワークライフバランスやキャリア成長など
  満足度面からの解釈もバランスよく含めること
- 断定しすぎず「〜の可能性が考えられます」のような表現を使うこと
- {tone}
- 出力は説明文のみ（見出しや箇条書きは不要）
- 太字（**）や見出し（#）などのMarkdown記法は使わず、プレーンテキストで出力すること"""

    client = get_client()
    response = client.messages.create(
        model=model,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def suggest_interventions(
    risk_pct: float,
    factors_text: str,
    explanation: str,
    model: str = MODEL_DEV,
) -> str:
    """⑤ 企業制度テキスト（簡易RAG）を踏まえて定着施策を提案する。"""
    policies_text = load_company_policies()
    tone = _INTERVENTION_TONE[get_risk_tier(risk_pct)]

    prompt = f"""あなたは人事コンサルタントです。以下の離職リスク分析結果と、
自社の制度一覧を踏まえて、この従業員の定着に向けた具体的な施策を2〜3個、
箇条書きで提案してください。

# 離職リスクスコア
{risk_pct:.1f}%

# 主要リスク要因
{factors_text}

# 原因分析結果
{explanation}

# 自社の制度一覧
{policies_text}

# 指示
- 必ず上記の制度一覧の中から、リスク要因に対応する制度を具体的に引用して提案すること
- 存在しない制度を創作しないこと
- 各提案は1〜2文程度で簡潔にまとめること
- {tone}
- 出力は箇条書きの施策のみ（前置きの挨拶などは不要）
- 太字（**）や見出し（#）などのMarkdown記法は使わず、プレーンテキストで出力すること"""

    client = get_client()
    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
