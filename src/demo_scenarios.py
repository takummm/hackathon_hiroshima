"""デモシナリオ構築（8/21タスク）: 高・中・低リスクの従業員を1名ずつ選定し、
④原因分析・⑤施策提案までの出力を確認し、data/demo_scenarios.txt に保存するスクリプト。

実行:
    venv/Scripts/python.exe src/demo_scenarios.py
"""

from pathlib import Path

from model import load_data, load_model, score_employee, top_risk_factors_for_employee, transform_with_encoders
from llm import MODEL_DEV, explain_risk_factors, suggest_interventions

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "demo_scenarios.txt"


def build_employee_summary(row) -> str:
    keys = [
        "Age", "Department", "JobRole", "MonthlyIncome", "DistanceFromHome",
        "OverTime", "JobSatisfaction", "WorkLifeBalance", "YearsAtCompany",
        "YearsSinceLastPromotion",
    ]
    return "\n".join(f"- {k}: {row[k]}" for k in keys if k in row.index)


def build_factors_text(factors_df) -> str:
    return "\n".join(
        f"- {r.feature}: importance={r.importance}, 従業員の値={r.employee_value}"
        for r in factors_df.itertuples()
    )


def pick_demo_employees(df, scores) -> dict:
    df = df.copy()
    df["risk_score"] = scores

    high = df.loc[df["risk_score"].idxmax()]
    low = df.loc[df["risk_score"].idxmin()]

    # 中リスク: 0.40〜0.60の範囲の中から中央値に最も近いものを選ぶ
    mid_range = df[(df["risk_score"] >= 0.40) & (df["risk_score"] <= 0.60)]
    if mid_range.empty:
        mid_range = df
    mid = mid_range.iloc[(mid_range["risk_score"] - 0.5).abs().argsort().iloc[0]]

    return {"高リスク": high, "中リスク": mid, "低リスク": low}


def main() -> None:
    model, encoders, feature_names, threshold = load_model()
    df = load_data()

    X_all = transform_with_encoders(df, encoders, feature_names)
    scores = model.predict_proba(X_all)[:, 1]

    employees = pick_demo_employees(df, scores)

    lines: list[str] = []
    lines.append("デモシナリオ: 高・中・低リスク従業員3名の分析結果")
    lines.append("=" * 70)
    lines.append("")

    for label, row in employees.items():
        emp_id = int(row["EmployeeNumber"])
        risk_proba = float(row["risk_score"])

        lines.append("=" * 70)
        lines.append(f"[{label}] EmployeeNumber={emp_id}  リスクスコア={risk_proba:.1%}  判定閾値={threshold:.0%}")
        lines.append("=" * 70)

        X_row = X_all.loc[[row.name]]
        factors = top_risk_factors_for_employee(model, X_row, top_n=5)
        factors_text = build_factors_text(factors)
        employee_summary = build_employee_summary(row)

        lines.append("--- feature importance ---")
        lines.append(factors_text)

        lines.append("\n--- ④ 原因分析 ---")
        explanation = explain_risk_factors(employee_summary, risk_proba * 100, factors_text, model=MODEL_DEV)
        lines.append(explanation)

        lines.append("\n--- ⑤ 施策提案 ---")
        interventions = suggest_interventions(risk_proba * 100, factors_text, explanation, model=MODEL_DEV)
        lines.append(interventions)
        lines.append("")

        print(f"[{label}] EmployeeNumber={emp_id} リスクスコア={risk_proba:.1%} 完了")

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n保存しました: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
