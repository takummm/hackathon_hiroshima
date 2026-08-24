"""広島県企業向け AI離職予防・人材定着支援サービス（MVP）

ステップ①〜⑤: CSV読込 → リスクスコア表示 → feature importance表示
→ ④LLMによる原因分析（自然文説明） → ⑤簡易RAGによる施策提案。
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from model import (  # noqa: E402
    DATA_PATH,
    ID_COLS,
    friendly_data_error_message,
    get_factor_risk_color,
    load_data,
    load_model,
    score_employee,
    top_risk_factors_for_employee,
    transform_with_encoders,
    validate_dataset,
)
from llm import (  # noqa: E402
    MODEL_DEV,
    MODEL_PROD,
    explain_risk_factors,
    friendly_error_message,
    get_risk_tier,
    suggest_interventions,
)
from labels import CURRENCY_COLS, format_value, to_ja  # noqa: E402

# Claude Designで作成したUI案（離職リスク分析ダッシュボード）の配色・タイポグラフィを移植。
RISK_TIER_COLOR = {"high": "#ff6b7a", "mid": "#f0c063", "low": "#6fe3a8"}

st.set_page_config(page_title="AI離職予防・人材定着支援", layout="wide")

FONT_AND_COLOR_CSS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+JP:wght@400;500;700;900&display=swap" rel="stylesheet">'
    "<style>"
    ":root {"
    "  --accent: #6ea8ff; --accent-soft: rgba(110, 168, 255, 0.10); --accent-border: rgba(110, 168, 255, 0.22);"
    "  --bg: #0a0d13; --surface: #111722; --surface-border: rgba(255, 255, 255, 0.07);"
    "}"
    'html, body, [class*="css"], .stButton button, .stSelectbox, .stTextInput,'
    ' [data-testid="stMetricLabel"], [data-testid="stMetricValue"], h1, h2, h3, h4,'
    ' [data-testid="stHeadingWithActionElements"],'
    ' [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] * {'
    '  font-family: "Inter", "Noto Sans JP", "Hiragino Sans", "Yu Gothic", sans-serif !important;'
    "}"
    '[data-testid="stAppViewContainer"], [data-testid="stSidebar"] { background: var(--bg) !important; }'
    ".st-key-risk-hero, .st-key-explanation-card, .st-key-interventions-card {"
    "  background: var(--surface) !important; border: 1px solid var(--surface-border) !important;"
    "  border-radius: 12px !important;"
    "}"
    "h1 { font-weight: 800 !important; letter-spacing: -0.02em; }"
    "h2, h3, h4 { font-weight: 700 !important; }"
    '[data-testid="stHeadingWithActionElements"] h3 {'
    "  border-left: 4px solid var(--accent) !important;"
    "  padding-left: 10px !important;"
    "  margin-top: 1.6rem !important;"
    "}"
    ".app-eyebrow {"
    "  color: var(--accent); font-size: 0.78rem; font-weight: 600; letter-spacing: 0.03em; margin-bottom: 4px;"
    "}"
    ".app-title { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; color: #f5f7fb; margin-bottom: 6px; }"
    ".app-desc { color: rgba(230, 235, 245, 0.65); font-size: 0.95rem; max-width: 62ch; line-height: 1.7; margin-bottom: 1.6rem; }"
    ".tech-badge {"
    "  display: inline-block; font-size: 0.72rem; font-weight: 600;"
    "  color: #8fb0ff; background: rgba(110, 168, 255, 0.1); border: 1px solid rgba(110, 168, 255, 0.2);"
    "  border-radius: 999px; padding: 3px 10px; margin: 2px 4px 2px 0;"
    "}"
    ".st-key-explanation-card p, .st-key-interventions-card p {"
    "  font-size: 1.05rem; line-height: 1.9; margin-bottom: 0.6rem;"
    "}"
    ".st-key-explanation-card, .st-key-interventions-card { padding: 1.1rem 1.3rem; }"
    ".ai-output-tag {"
    "  display: inline-block; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.03em;"
    "  color: var(--accent); background: var(--accent-soft); border: 1px solid var(--accent-border);"
    "  border-radius: 999px; padding: 3px 10px; margin-bottom: 10px;"
    "}"
    ".stButton button {"
    "  background: #161d2b !important; border: 1px solid rgba(255, 255, 255, 0.1) !important;"
    "  border-radius: 10px !important; font-weight: 600 !important;"
    "}"
    ".stButton button:hover { border-color: var(--accent) !important; color: var(--accent) !important; }"
    "</style>"
)
st.markdown(FONT_AND_COLOR_CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="app-eyebrow">AI離職予防・人材定着支援サービス</div>'
    '<div class="app-title">従業員リスク分析ダッシュボード</div>'
    '<div class="app-desc">CSVから従業員データを読み込み、離職リスクをスコア化。'
    "主要因を提示し、Claude APIが原因分析と定着施策を自然文で提案します。</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("LLM設定")
    is_prod_mode = st.toggle("デモ本番モード（Sonnet 5使用）", value=False)
    llm_model = MODEL_PROD if is_prod_mode else MODEL_DEV
    st.caption(f"使用モデル: {llm_model}")
    st.markdown(
        '<div style="margin-top:1.8rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.07);">'
        '<span class="tech-badge">LightGBM</span>'
        '<span class="tech-badge">Claude API</span>'
        '<span class="tech-badge">Streamlit</span>'
        "</div>",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="🧠 学習済みモデルを読み込んでいます…")
def get_model():
    return load_model()


try:
    model, encoders, feature_names, threshold = get_model()
except FileNotFoundError:
    st.error(
        "学習済みモデルが見つかりません。先に `venv/Scripts/python.exe src/train_model.py` "
        "を実行してモデルを学習・保存してください。"
    )
    st.stop()


@st.cache_data(show_spinner="📄 従業員データを読み込んでいます…")
def get_dataset():
    return load_data(DATA_PATH)


df = get_dataset()

dataset_issues = validate_dataset(df, feature_names)
if dataset_issues:
    st.error(
        "⚠️ 読み込んだCSVに問題があり、正しく分析できません。\n\n"
        + "\n".join(f"- {issue}" for issue in dataset_issues)
    )
    st.stop()

with st.expander(f"① 従業員データ（CSV）を表示 — {DATA_PATH.name}（{len(df)}件）", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)


@st.cache_resource(show_spinner="📊 デモ用の高・中・低リスク従業員を選定しています…")
def get_demo_employees() -> dict:
    """デモ用に高・中・低リスクの従業員を1名ずつ選定する（data/demo_scenarios.txtと同じロジック）。"""
    X_all = transform_with_encoders(df, encoders, feature_names)
    scores = model.predict_proba(X_all)[:, 1]
    scored = df.copy()
    scored["risk_score"] = scores

    high = scored.loc[scored["risk_score"].idxmax()]
    low = scored.loc[scored["risk_score"].idxmin()]
    mid_range = scored[(scored["risk_score"] >= 0.40) & (scored["risk_score"] <= 0.60)]
    if mid_range.empty:
        mid_range = scored
    mid = mid_range.iloc[(mid_range["risk_score"] - 0.5).abs().argsort().iloc[0]]

    return {
        "高リスク": int(high["EmployeeNumber"]),
        "中リスク": int(mid["EmployeeNumber"]),
        "低リスク": int(low["EmployeeNumber"]),
    }


demo_employees = get_demo_employees()

st.subheader("従業員を選択")
st.caption("デモ用ショートカット（高・中・低リスクの代表例）")
demo_cols = st.columns(3)
for demo_col, (demo_label, demo_emp_id) in zip(demo_cols, demo_employees.items()):
    if demo_col.button(f"{demo_label}の例（#{demo_emp_id}）"):
        st.session_state["employee_select"] = demo_emp_id

employee_ids = df["EmployeeNumber"].tolist()
selected_id = st.selectbox("EmployeeNumber", employee_ids, key="employee_select")

employee_row = df[df["EmployeeNumber"] == selected_id]

try:
    X_row = transform_with_encoders(employee_row, encoders, feature_names)
    risk_proba = score_employee(model, X_row)
except (KeyError, ValueError, TypeError) as e:
    st.error(f"⚠️ {friendly_data_error_message(e)}")
    st.stop()

is_high_risk = risk_proba >= threshold
risk_tier = get_risk_tier(risk_proba * 100)
risk_color = RISK_TIER_COLOR[risk_tier]

st.subheader("📊 リスク分析結果")
with st.container(border=True, key="risk-hero"):
    st.markdown(
        f"""
        <div style="background:{risk_color}14;border-radius:8px;padding:4px 0 0;">
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div style="font-size:0.9rem;color:rgba(150,150,150,0.9);margin-bottom:4px;">離職リスクスコア</div>
            <div style="font-size:3rem;font-weight:800;color:{risk_color};line-height:1.15;">{risk_proba:.1%}</div>
            """,
            unsafe_allow_html=True,
        )
    col2.metric("判定", "⚠️ 要注意" if is_high_risk else "OK")
    st.markdown("</div>", unsafe_allow_html=True)

st.subheader("② 従業員プロフィール")
st.caption("💵 給与関連の項目は米ドル(USD)建てです（元データセットの通貨単位に準拠）。")
show_cols = [
    c for c in df.columns if c not in ID_COLS and c != "Attrition"
]
profile_row = employee_row.iloc[0]
profile_table = pd.DataFrame(
    {"値": [format_value(c, profile_row[c]) for c in show_cols]},
    index=[to_ja(c) for c in show_cols],
)
st.dataframe(profile_table, use_container_width=True)

st.subheader("③ 主要リスク要因")
factors = top_risk_factors_for_employee(model, X_row, top_n=5)
factors_display = factors.copy()
factors_display.insert(0, "要因（日本語）", factors_display["feature"].apply(to_ja))
factors_display["この従業員の値"] = factors_display.apply(
    lambda r: format_value(r["feature"], r["employee_value"]), axis=1
)

# 全体平均との比較を追加
def compute_pct_diff(row):
    feature = row["feature"]
    if feature not in df.columns:
        return "（N/A）"
    try:
        avg = float(df[feature].mean())
        val = float(row["employee_value"])
        if avg == 0:
            return "（N/A）"
        pct = ((val - avg) / abs(avg)) * 100
        sign = "+" if pct > 0 else ""
        return f"{sign}{pct:.0f}%"
    except Exception:
        return "（N/A）"

def compute_risk_badge(row):
    try:
        color = get_factor_risk_color(row["feature"], float(row["employee_value"]), df)
        emoji = {"high": "🔴", "mid": "🟡", "low": "🟢"}.get(color, "⚪")
        label = {"high": "高", "mid": "中", "low": "低"}.get(color, "不明")
        return f"{emoji} {label}"
    except Exception:
        return "⚪ 不明"

factors_display["平均との比較"] = factors_display.apply(compute_pct_diff, axis=1)
factors_display["リスク度"] = factors_display.apply(compute_risk_badge, axis=1)

factors_display = factors_display.rename(
    columns={"feature": "列名（英語）", "importance": "重要度"}
)[["要因（日本語）", "列名（英語）", "重要度", "この従業員の値", "平均との比較", "リスク度"]]

st.markdown("### 📊 主要リスク要因の詳細（平均値比較 + リスク度）")
st.dataframe(factors_display, use_container_width=True, hide_index=True)

st.markdown("### 重要度グラフ")
st.bar_chart(factors_display.set_index("要因（日本語）")["重要度"])


def build_employee_summary(row: pd.Series) -> str:
    keys = [
        "Age", "Department", "JobRole", "MonthlyIncome", "DistanceFromHome",
        "OverTime", "JobSatisfaction", "WorkLifeBalance", "YearsAtCompany",
        "YearsSinceLastPromotion",
    ]
    return "\n".join(f"- {k}: {row[k]}" for k in keys if k in row)


def build_factors_text(factors_df: pd.DataFrame) -> str:
    return "\n".join(
        f"- {r.feature}: importance={r.importance}, 従業員の値={r.employee_value}"
        for r in factors_df.itertuples()
    )


employee_summary = build_employee_summary(employee_row.iloc[0])
factors_text = build_factors_text(factors)


@st.cache_data(show_spinner=False)
def cached_explain(emp_id: int, risk_pct: float, factors_text: str, employee_summary: str, model: str) -> str:
    return explain_risk_factors(employee_summary, risk_pct, factors_text, model=model)


@st.cache_data(show_spinner=False)
def cached_suggest(emp_id: int, risk_pct: float, factors_text: str, explanation: str, model: str) -> str:
    return suggest_interventions(risk_pct, factors_text, explanation, model=model)


explanation_key = f"explanation_{selected_id}"
interventions_key = f"interventions_{selected_id}"

st.subheader("④ 原因分析（LLM）")
if st.button("原因分析を実行"):
    try:
        with st.spinner(f"🔍 {llm_model} が離職リスクの要因を分析しています…（数秒お待ちください）"):
            st.session_state[explanation_key] = cached_explain(
                selected_id, risk_proba * 100, factors_text, employee_summary, llm_model
            )
    except Exception as e:  # noqa: BLE001
        st.error(f"⚠️ {friendly_error_message(e)}")

if explanation_key in st.session_state:
    with st.container(border=True, key="explanation-card"):
        st.markdown(f'<span class="ai-output-tag">🤖 {llm_model} 生成</span>', unsafe_allow_html=True)
        st.markdown(st.session_state[explanation_key])

st.subheader("⑤ 定着施策の提案（簡易RAG）")
if st.button("施策提案を実行"):
    if explanation_key not in st.session_state:
        st.warning("先に「原因分析を実行」してください。")
    else:
        try:
            with st.spinner("📋 自社の制度と照らし合わせて定着施策を検討しています…（数秒お待ちください）"):
                st.session_state[interventions_key] = cached_suggest(
                    selected_id,
                    risk_proba * 100,
                    factors_text,
                    st.session_state[explanation_key],
                    llm_model,
                )
        except Exception as e:  # noqa: BLE001
            st.error(f"⚠️ {friendly_error_message(e)}")

if interventions_key in st.session_state:
    with st.container(border=True, key="interventions-card"):
        st.markdown(f'<span class="ai-output-tag">🤖 {llm_model} 生成</span>', unsafe_allow_html=True)
        items = [p.strip() for p in st.session_state[interventions_key].split("\n\n") if p.strip()]
        for i, item in enumerate(items):
            st.markdown(item)
            if i < len(items) - 1:
                st.divider()
