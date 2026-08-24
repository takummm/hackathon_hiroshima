"""IBM HR Analytics データセットの列名を、企業担当者向けの日本語表示に変換するための辞書。

UI表示（feature importance・従業員プロフィール等）専用。モデル学習・LLMプロンプトの
内部処理では英語の列名をそのまま使う（学習済みモデルとの整合性を保つため）。
"""

FEATURE_LABELS_JA: dict[str, str] = {
    "Age": "年齢",
    "Attrition": "離職有無",
    "BusinessTravel": "出張頻度",
    "DailyRate": "日給(USD)",
    "Department": "部門",
    "DistanceFromHome": "通勤距離(km)",
    "Education": "学歴レベル",
    "EducationField": "専攻分野",
    "EmployeeCount": "従業員数(定数)",
    "EmployeeNumber": "従業員番号",
    "EnvironmentSatisfaction": "職場環境満足度",
    "Gender": "性別",
    "HourlyRate": "時給(USD)",
    "JobInvolvement": "仕事への関与度",
    "JobLevel": "職位レベル",
    "JobRole": "職種",
    "JobSatisfaction": "仕事満足度",
    "MaritalStatus": "婚姻状況",
    "MonthlyIncome": "月収(USD)",
    "MonthlyRate": "月額レート(USD)",
    "NumCompaniesWorked": "経験企業数",
    "Over18": "成人フラグ",
    "OverTime": "残業有無",
    "PercentSalaryHike": "昇給率(%)",
    "PerformanceRating": "人事評価",
    "RelationshipSatisfaction": "人間関係満足度",
    "StandardHours": "標準労働時間",
    "StockOptionLevel": "ストックオプション水準",
    "TotalWorkingYears": "総勤務年数",
    "TrainingTimesLastYear": "昨年の研修回数",
    "WorkLifeBalance": "ワークライフバランス",
    "YearsAtCompany": "勤続年数",
    "YearsInCurrentRole": "現職の在任年数",
    "YearsSinceLastPromotion": "前回昇進からの年数",
    "YearsWithCurrManager": "現上司との年数",
}


def to_ja(feature_name: str) -> str:
    """英語の列名を日本語表示名に変換する。未登録の列名はそのまま返す。"""
    return FEATURE_LABELS_JA.get(feature_name, feature_name)


# IBM HR Analyticsデータセットは米ドル(USD)建ての金額列を含む（公式ドキュメント上、通貨単位の
# 明記はないが、原データセットの慣例としてUSD前提で扱われることが一般的なため、その旨を明示する）。
CURRENCY_COLS = {"MonthlyIncome", "DailyRate", "MonthlyRate", "HourlyRate"}


def format_value(feature_name: str, value) -> str:
    """表示用に値をフォーマットする。通貨列は $記号+桁区切りを付与する。"""
    if feature_name in CURRENCY_COLS:
        try:
            return f"${float(value):,.0f}"
        except (TypeError, ValueError):
            return str(value)
    return str(value)
