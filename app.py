"""広島県企業向け AI離職予防・人材定着支援サービス（MVP）

タブ構成:
  タブ1: 全体リスク一覧 & ターゲット選定（CSV確認・組織全体のリスク一覧・従業員選択）
  タブ2: 個別詳細カルテ（リスクスコア・プロフィール・主要リスク要因）
  タブ3: AIアクションプラン（LLMによる原因分析・簡易RAGによる施策提案）
"""

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from model import (  # noqa: E402
    DATA_PATH,
    ID_COLS,
    friendly_data_error_message,
    load_data,
    load_model,
    risk_contributions_for_employee,
    score_employee,
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
from labels import CURRENCY_COLS, format_value, to_ja, to_ja_value  # noqa: E402
from colors import RISK_TIER_COLOR  # noqa: E402

# ライトテーマ（濃紺・ネイビー基調 + 青系アクセント）の配色・タイポグラフィ。
RISK_TIER_BADGE = {"high": "🔴 高", "mid": "🟡 中", "low": "🟢 低"}

st.set_page_config(page_title="AI離職予防・人材定着支援", layout="wide")

FONT_AND_COLOR_CSS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+JP:wght@400;500;700;900&display=swap" rel="stylesheet">'
    "<style>"
    ":root {"
    "  --accent: #2563eb; --accent-soft: rgba(37, 99, 235, 0.08); --accent-border: rgba(37, 99, 235, 0.25);"
    "  --navy: #1e3a8a;"
    "  --bg: #f4f6f9; --surface: #ffffff; --surface-border: #e2e8f0;"
    "  --text: #1e293b; --text-dim: #475569; --text-faint: #94a3b8;"
    "}"
    'html, body, [class*="css"], .stButton button, .stSelectbox, .stTextInput,'
    ' [data-testid="stMetricLabel"], [data-testid="stMetricValue"], h1, h2, h3, h4,'
    ' [data-testid="stHeadingWithActionElements"],'
    ' [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] * {'
    '  font-family: "Inter", "Noto Sans JP", "Hiragino Sans", "Yu Gothic", sans-serif !important;'
    "}"
    ".st-key-risk-hero, .st-key-explanation-card, .st-key-interventions-card {"
    "  background: var(--surface) !important; border: 1px solid var(--surface-border) !important;"
    "  border-radius: 10px !important;"
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
    ".app-title { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; color: var(--navy); margin-bottom: 6px; }"
    ".app-desc { color: var(--text-dim); font-size: 0.95rem; max-width: 62ch; line-height: 1.7; margin-bottom: 1.6rem; }"
    ".tech-badge {"
    "  display: inline-block; font-size: 0.72rem; font-weight: 600;"
    "  color: #1d4ed8; background: rgba(37, 99, 235, 0.08); border: 1px solid rgba(37, 99, 235, 0.2);"
    "  border-radius: 999px; padding: 3px 10px; margin: 2px 4px 2px 0;"
    "}"
    ".st-key-explanation-card p, .st-key-interventions-card p {"
    "  font-size: 1.05rem; line-height: 1.9; margin-bottom: 0.6rem; color: var(--text);"
    "}"
    ".st-key-explanation-card, .st-key-interventions-card { padding: 1.1rem 1.3rem; }"
    ".ai-output-tag {"
    "  display: inline-block; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.03em;"
    "  color: var(--accent); background: var(--accent-soft); border: 1px solid var(--accent-border);"
    "  border-radius: 999px; padding: 3px 10px; margin-bottom: 10px;"
    "}"
    ".stButton button {"
    "  background: #f8fafc !important; border: 1px solid var(--surface-border) !important;"
    "  border-radius: 8px !important; font-weight: 600 !important; color: var(--text) !important;"
    "}"
    ".stButton button:hover { border-color: var(--accent) !important; color: var(--accent) !important; }"
    '.stTabs [data-baseweb="tab-list"] { gap: 4px; }'
    '.stTabs [data-baseweb="tab"] { font-weight: 600 !important; }'
    "</style>"
)
st.markdown(FONT_AND_COLOR_CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="app-eyebrow">AI離職予防・人材定着支援サービス</div>'
    '<div class="app-title">従業員リスク分析ダッシュボード</div>'
    '<div class="app-desc">従業員データを読み込み、離職リスクをスコア化。'
    "要因を示し、AIが原因分析と定着施策を提案します。</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("LLM設定")
    is_prod_mode = st.toggle("デモ本番モード（Sonnet 5使用）", value=False)
    llm_model = MODEL_PROD if is_prod_mode else MODEL_DEV
    st.caption(f"使用モデル: {llm_model}")
    st.markdown(
        '<div style="margin-top:1.8rem;padding-top:1rem;border-top:1px solid #e2e8f0;">'
        '<span class="tech-badge">LightGBM</span>'
        '<span class="tech-badge">Claude API</span>'
        '<span class="tech-badge">Streamlit</span>'
        "</div>",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="学習済みモデルを読み込んでいます…")
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


@st.cache_data(show_spinner="従業員データを読み込んでいます…")
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


@st.cache_data(show_spinner="全従業員のリスクスコアを計算しています…")
def get_scored_employees() -> pd.DataFrame:
    """全従業員分のリスクスコアを一括計算する（全体リスク一覧・デモ選定の両方で使用）。"""
    X_all = transform_with_encoders(df, encoders, feature_names)
    scores = model.predict_proba(X_all)[:, 1]
    scored = df.copy()
    scored["risk_score"] = scores
    return scored


scored_df = get_scored_employees()


def pick_demo_employees(scored: pd.DataFrame) -> dict:
    """デモ用に高・中・低リスクの従業員を1名ずつ選定する（data/demo_scenarios.txtと同じロジック）。"""
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


demo_employees = pick_demo_employees(scored_df)

tab1, tab2, tab3 = st.tabs(
    ["全体リスク一覧 & ターゲット選定", "個別詳細カルテ", "AIによる提案"]
)

with tab1:
    with st.expander(f"元データ（CSV）を表示 — {DATA_PATH.name}（{len(df)}件）", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    st.subheader("部署別リスクヒートマップ")
    st.caption("部門×職種ごとの平均離職リスク。色が濃いほどリスクが高い組み合わせです。")

    heatmap_df = scored_df[["Department", "JobRole", "risk_score"]].copy()
    heatmap_df["部門"] = heatmap_df["Department"].apply(lambda v: to_ja_value("Department", v))
    heatmap_df["職種"] = heatmap_df["JobRole"].apply(lambda v: to_ja_value("JobRole", v))
    heatmap_agg = heatmap_df.groupby(["部門", "職種"], as_index=False).agg(
        平均リスク=("risk_score", "mean"), 人数=("risk_score", "size")
    )
    heatmap_agg["平均リスク(%)"] = (heatmap_agg["平均リスク"] * 100).round(1)

    heatmap_base = alt.Chart(heatmap_agg).encode(
        x=alt.X("部門:N", title=None, axis=alt.Axis(labelAngle=0, labelLimit=220)),
        y=alt.Y("職種:N", title=None, axis=alt.Axis(labelLimit=220)),
    )
    heatmap_max = max(heatmap_agg["平均リスク(%)"].max(), 1.0)
    heatmap_rect = heatmap_base.mark_rect().encode(
        color=alt.Color(
            "平均リスク(%):Q",
            scale=alt.Scale(
                domain=[0, heatmap_max / 2, heatmap_max],
                range=[RISK_TIER_COLOR["low"], RISK_TIER_COLOR["mid"], RISK_TIER_COLOR["high"]],
            ),
            title="平均リスク(%)",
        ),
        tooltip=[
            "部門",
            "職種",
            alt.Tooltip("平均リスク(%):Q", format=".1f"),
            alt.Tooltip("人数:Q", title="人数"),
        ],
    )
    heatmap_text = heatmap_base.mark_text(baseline="middle", fontSize=12, color="#1e293b").encode(
        text=alt.Text("平均リスク(%):Q", format=".0f"),
    )
    st.altair_chart(
        (heatmap_rect + heatmap_text).properties(height=340),
        use_container_width=True,
    )

    st.subheader("組織全体のリスク一覧")
    st.caption("離職リスクスコアが高い順に表示しています。検索・列並び替えが可能です。")

    org_table = scored_df[["EmployeeNumber", "Department", "JobRole", "risk_score"]].copy()
    org_table = org_table.sort_values("risk_score", ascending=False).reset_index(drop=True)
    org_table["Department"] = org_table["Department"].apply(lambda v: to_ja_value("Department", v))
    org_table["JobRole"] = org_table["JobRole"].apply(lambda v: to_ja_value("JobRole", v))
    org_table["リスクスコア"] = (org_table["risk_score"] * 100).round(1).astype(str) + "%"
    org_table["リスク度"] = org_table["risk_score"].apply(
        lambda s: RISK_TIER_BADGE[get_risk_tier(s * 100)]
    )
    org_table_display = org_table.rename(
        columns={"EmployeeNumber": "従業員番号", "Department": "部門", "JobRole": "職種"}
    )[["従業員番号", "部門", "職種", "リスクスコア", "リスク度"]]
    st.dataframe(org_table_display, use_container_width=True, hide_index=True, height=320)

    high_count = int((scored_df["risk_score"] * 100 >= 70).sum())
    mid_count = int(
        ((scored_df["risk_score"] * 100 < 70) & (scored_df["risk_score"] * 100 > 10)).sum()
    )
    low_count = int((scored_df["risk_score"] * 100 <= 10).sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("🔴 高リスク", f"{high_count}名")
    m2.metric("🟡 中リスク", f"{mid_count}名")
    m3.metric("🟢 低リスク", f"{low_count}名")

    st.divider()
    st.subheader("ターゲット選定")
    st.caption("デモ用ショートカット（高・中・低リスクの代表例）")
    demo_cols = st.columns(3)
    for demo_col, (demo_label, demo_emp_id) in zip(demo_cols, demo_employees.items()):
        if demo_col.button(f"{demo_label}の例（#{demo_emp_id}）"):
            st.session_state["employee_select"] = demo_emp_id

    employee_ids = df["EmployeeNumber"].tolist()
    selected_id = st.selectbox("EmployeeNumber", employee_ids, key="employee_select")

# --- ここから選択中の従業員に対する分析（②③タブで使用） ---
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

with tab2:
    st.caption(f"選択中の従業員: **#{selected_id}**（タブ1で変更できます）")

    st.subheader("リスク分析結果")
    # ごく小さい非ゼロのスコアが .1% 丸めで「0.0%」と表示され、
    # リスクが完全に無いかのように誤認されるのを防ぐ。
    risk_score_text = "0.1%未満" if 0 < risk_proba < 0.001 else f"{risk_proba:.1%}"
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
                <div style="font-size:0.9rem;color:#64748b;margin-bottom:4px;" title="過去の従業員データから学習したAIモデルが、給与・勤務状況・満足度など複数の要因をもとに算出した離職の可能性です。100%に近いほど、過去に離職した人と似た傾向が強いことを意味します（モデルの的中率とは別の指標です）。">離職リスクスコア ⓘ</div>
                <div style="font-size:3rem;font-weight:800;color:{risk_color};line-height:1.15;">{risk_score_text}</div>
                """,
                unsafe_allow_html=True,
            )
        col2.metric(
            "判定",
            "⚠️ 要注意" if is_high_risk else "問題なし",
            help="離職リスクスコアが30%を超えると「要注意」と判定します。離職の兆候をできるだけ早く察知するために設定した基準です。",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    profile_col, factors_col = st.columns([2, 3])

    with profile_col:
        st.subheader("従業員プロフィール")
        st.caption(
            "給与関連の項目は米ドル(USD)建てです（元データセットの通貨単位に準拠）。"
            "円換算は固定レート（1ドル=150円）による概算です。"
        )
        show_cols = [c for c in df.columns if c not in ID_COLS and c != "Attrition"]
        profile_row = employee_row.iloc[0]
        profile_table = pd.DataFrame(
            {"値": [format_value(c, profile_row[c]) for c in show_cols]},
            index=[to_ja(c) for c in show_cols],
        )
        st.dataframe(profile_table, use_container_width=True, height=460)

    with factors_col:
        st.subheader("主要リスク要因")
        factors = risk_contributions_for_employee(model, X_row, employee_row, df, top_n=6)

        factors_view = factors.copy()
        factors_view["要因"] = factors_view["feature"].apply(to_ja)
        factors_view["値"] = factors_view.apply(
            lambda r: format_value(r["feature"], r["employee_value"]), axis=1
        )
        factors_view["方向"] = factors_view["contribution"].apply(
            lambda c: "リスクを高める" if c > 0 else "リスクを下げる"
        )
        factors_view["ラベル"] = factors_view["要因"] + "（" + factors_view["値"].astype(str) + "）"

        def compute_avg_diff(row):
            avg = row["company_avg"]
            if avg is None or pd.isna(avg) or avg == 0:
                return "—"
            try:
                pct = ((float(row["employee_value"]) - avg) / abs(avg)) * 100
                return f"{'+' if pct > 0 else ''}{pct:.0f}%"
            except (TypeError, ValueError):
                return "—"

        factors_view["全社平均"] = factors_view.apply(
            lambda r: "—"
            if r["company_avg"] is None or pd.isna(r["company_avg"])
            else format_value(r["feature"], r["company_avg"]),
            axis=1,
        )
        factors_view["平均との差"] = factors_view.apply(compute_avg_diff, axis=1)

        st.markdown("##### 離職リスクへの影響")

        # モデルが算出した寄与度(log-odds空間)そのままだと目盛りの数値が直感的でないため、
        # 最大の影響度を100とした相対スケールに正規化して表示する（符号＝方向は保持）。
        max_abs = factors_view["contribution"].abs().max() or 1.0
        factors_view["影響度"] = (factors_view["contribution"] / max_abs * 100).round(1)

        chart_df = factors_view.sort_values("contribution")
        bar_order = chart_df["ラベル"].tolist()
        has_both_directions = (
            bool((chart_df["contribution"] > 0).any())
            and bool((chart_df["contribution"] < 0).any())
        )

        impact_bars = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("影響度:Q", title="離職リスクへの影響度"),
            y=alt.Y("ラベル:N", sort=bar_order, title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.Color(
                "方向:N",
                scale=alt.Scale(
                    domain=["リスクを高める", "リスクを下げる"],
                    range=[RISK_TIER_COLOR["high"], RISK_TIER_COLOR["low"]],
                ),
                legend=alt.Legend(title=None, orient="top") if has_both_directions else None,
            ),
            tooltip=[
                alt.Tooltip("要因:N"),
                alt.Tooltip("値:N", title="この従業員の値"),
                alt.Tooltip("影響度:Q", title="影響度"),
            ],
        )
        zero_rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#94a3b8").encode(x="x:Q")
        st.altair_chart((impact_bars + zero_rule).properties(height=260), use_container_width=True)

        st.markdown("##### 本人と全社平均の比較")
        table_df = factors_view.rename(columns={"値": "この従業員の値"})[
            ["要因", "この従業員の値", "全社平均", "平均との差"]
        ]
        st.dataframe(table_df, use_container_width=True, hide_index=True, height=250)


def build_employee_summary(row: pd.Series) -> str:
    """LLMプロンプト用の従業員概要を日本語で構築する。
    英語の列名をそのまま渡すと、LLMの生成文にも英語が混入することがあるため、
    ラベル・値ともに日本語表記に変換してから渡す。"""
    keys = [
        "Age", "Department", "JobRole", "MonthlyIncome", "DistanceFromHome",
        "OverTime", "JobSatisfaction", "WorkLifeBalance", "YearsAtCompany",
        "YearsSinceLastPromotion",
    ]
    return "\n".join(f"- {to_ja(k)}: {format_value(k, row[k])}" for k in keys if k in row)


def build_factors_text(factors_df: pd.DataFrame) -> str:
    """LLMプロンプト用の主要リスク要因を日本語で構築する（理由はbuild_employee_summaryと同様）。

    contribution の符号で、その要因が離職リスクを高めているか下げているかを併記する。"""
    lines = []
    for r in factors_df.itertuples():
        direction = "離職リスクを高める方向" if r.contribution > 0 else "定着に寄与する方向"
        lines.append(
            f"- {to_ja(r.feature)}: 従業員の値={format_value(r.feature, r.employee_value)}（{direction}）"
        )
    return "\n".join(lines)


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

with tab3:
    st.caption(f"選択中の従業員: **#{selected_id}**（タブ1で変更できます）")

    st.subheader("原因分析")

    if st.button("原因分析を実行"):
        try:
            with st.spinner(f"{llm_model} が離職リスクの要因を分析しています…（数秒お待ちください）"):
                st.session_state[explanation_key] = cached_explain(
                    selected_id, risk_proba * 100, factors_text, employee_summary, llm_model
                )
        except Exception as e:  # noqa: BLE001
            st.error(f"⚠️ {friendly_error_message(e)}")

    if explanation_key in st.session_state:
        with st.container(border=True, key="explanation-card"):
            st.markdown(st.session_state[explanation_key])

    st.subheader("定着施策の提案")
    if st.button("施策提案を実行"):
        if explanation_key not in st.session_state:
            st.warning("先に「原因分析を実行」してください。")
        else:
            try:
                with st.spinner("自社の制度と照らし合わせて定着施策を検討しています…（数秒お待ちください）"):
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
            items = [p.strip() for p in st.session_state[interventions_key].split("\n\n") if p.strip()]
            for i, item in enumerate(items):
                st.markdown(item)
                if i < len(items) - 1:
                    st.divider()
