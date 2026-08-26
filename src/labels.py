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

# 円換算用の固定レート。このデータセットは特定の年月日に紐づかない架空データのため、
# リアルタイムAPI取得はせず、概算表示用の固定値を用いる（表示層のみで使用。元データ・
# モデル学習・LLMプロンプトには影響しない）。
USD_TO_JPY_RATE = 150


# カテゴリ列の値を日本語表示に変換するための辞書（列名 → {英語値: 日本語値}）。
# labels.pyの方針と同様、モデル学習・LLMプロンプトの内部処理では英語の値をそのまま使う。
CATEGORY_LABELS_JA: dict[str, dict[str, str]] = {
    "Attrition": {"No": "在籍", "Yes": "離職"},
    "BusinessTravel": {
        "Non-Travel": "出張なし",
        "Travel_Rarely": "稀に出張",
        "Travel_Frequently": "頻繁に出張",
    },
    "Department": {
        "Human Resources": "人事部",
        "Research & Development": "研究開発部",
        "Sales": "営業部",
    },
    "EducationField": {
        "Human Resources": "人事",
        "Life Sciences": "生命科学",
        "Marketing": "マーケティング",
        "Medical": "医療",
        "Other": "その他",
        "Technical Degree": "技術系",
    },
    "Gender": {"Female": "女性", "Male": "男性"},
    "JobRole": {
        "Healthcare Representative": "ヘルスケア担当",
        "Human Resources": "人事",
        "Laboratory Technician": "検査技師",
        "Manager": "マネージャー",
        "Manufacturing Director": "製造部門責任者",
        "Research Director": "研究部門責任者",
        "Research Scientist": "研究員",
        "Sales Executive": "営業（エグゼクティブ）",
        "Sales Representative": "営業（担当者）",
    },
    "MaritalStatus": {"Divorced": "離婚", "Married": "既婚", "Single": "未婚"},
    "Over18": {"Y": "はい"},
    "OverTime": {"No": "なし", "Yes": "あり"},
}


def to_ja_value(feature_name: str, value) -> str:
    """カテゴリ値を日本語表示に変換する。未登録の列・値はそのまま返す。"""
    mapping = CATEGORY_LABELS_JA.get(feature_name)
    if mapping is None:
        return str(value)
    return mapping.get(str(value), str(value))


# 順序尺度（数値の大小に意味がある列）を、IBM HR Analyticsデータセットの公式ドキュメントに
# 記載された尺度定義に基づいて日本語ラベル化する。JobLevel・StockOptionLevel等は
# 公式な尺度定義が公開されていないため、意味を捏造しないよう対象外とする。
ORDINAL_LABELS_JA: dict[str, dict[int, str]] = {
    "Education": {
        1: "高校卒以下",
        2: "短大・専門卒",
        3: "大学卒",
        4: "大学院卒（修士）",
        5: "大学院卒（博士）",
    },
    "EnvironmentSatisfaction": {1: "低い", 2: "普通", 3: "高い", 4: "非常に高い"},
    "JobInvolvement": {1: "低い", 2: "普通", 3: "高い", 4: "非常に高い"},
    "JobSatisfaction": {1: "低い", 2: "普通", 3: "高い", 4: "非常に高い"},
    "PerformanceRating": {1: "要改善", 2: "良好", 3: "優秀", 4: "卓越"},
    "RelationshipSatisfaction": {1: "低い", 2: "普通", 3: "高い", 4: "非常に高い"},
    "WorkLifeBalance": {1: "悪い", 2: "良い", 3: "より良い", 4: "最良"},
}


def to_ja_ordinal(feature_name: str, value) -> str | None:
    """順序尺度の値を「ラベル (値/最大値)」形式に変換する。対象外の列はNoneを返す。"""
    mapping = ORDINAL_LABELS_JA.get(feature_name)
    if mapping is None:
        return None
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    label = mapping.get(v)
    if label is None:
        return None
    max_v = max(mapping.keys())
    return f"{label} ({v}/{max_v})"


def format_value(feature_name: str, value) -> str:
    """表示用に値をフォーマットする。通貨列は $記号+桁区切り+円換算（固定レート）、
    カテゴリ列・順序尺度列は日本語表記に変換する。"""
    if feature_name in CURRENCY_COLS:
        try:
            usd = float(value)
            jpy_man = usd * USD_TO_JPY_RATE / 10000
            return f"${usd:,.0f}（約{jpy_man:.1f}万円）"
        except (TypeError, ValueError):
            return str(value)
    if feature_name in CATEGORY_LABELS_JA:
        return to_ja_value(feature_name, value)
    ordinal = to_ja_ordinal(feature_name, value)
    if ordinal is not None:
        return ordinal
    return str(value)
