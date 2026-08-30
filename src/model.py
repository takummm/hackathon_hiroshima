"""離職リスク予測モデル（ステップ②③）: 学習・評価・feature importance算出、
および個々の従業員に対するスコア算出・要因抽出のユーティリティ。"""

from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

TARGET_COL = "Attrition"
ID_COLS = ["EmployeeNumber"]
CONSTANT_COLS = ["EmployeeCount", "StandardHours", "Over18"]

# 仕様書の方針「見逃さない（Recall優先）」に合わせたデフォルト判定閾値。
# 0.5だとRecallが低すぎる（離職者を見逃す）ため引き下げている。
DEFAULT_THRESHOLD = 0.30


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_dataset(df: pd.DataFrame, feature_names: list[str]) -> list[str]:
    """CSVの想定外データ（列欠落・全欠損列など）を検出し、問題点のリストを返す。
    空リストなら問題なし。"""
    issues: list[str] = []

    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        issues.append(f"必要な列が見つかりません: {', '.join(missing_cols)}")

    if "EmployeeNumber" not in df.columns:
        issues.append("従業員を一意に識別する 'EmployeeNumber' 列が見つかりません。")

    all_null_cols = [c for c in feature_names if c in df.columns and df[c].isna().all()]
    if all_null_cols:
        issues.append(f"全ての値が欠損している列があります: {', '.join(all_null_cols)}")

    return issues


def friendly_data_error_message(e: Exception) -> str:
    """従業員データのスコア計算失敗時に、ユーザー向けの分かりやすいメッセージへ変換する。"""
    if isinstance(e, KeyError):
        return "この従業員のデータに必要な項目が不足しているため、分析できませんでした。CSVの列構成を確認してください。"
    if isinstance(e, (ValueError, TypeError)):
        return "この従業員のデータに欠損値または想定外の形式が含まれているため、分析できませんでした。CSVの内容を確認してください。"
    return "この従業員のデータを処理できませんでした。CSVの内容を確認するか、別の従業員を選択してください。"


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, LabelEncoder]]:
    """特徴量Xとターゲットyに分割し、カテゴリ列をLabelEncoderで数値化する。"""
    df = df.drop(columns=[c for c in ID_COLS + CONSTANT_COLS if c in df.columns])

    y = (df[TARGET_COL] == "Yes").astype(int)
    X = df.drop(columns=[TARGET_COL])

    encoders: dict[str, LabelEncoder] = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    return X, y, encoders


def transform_with_encoders(
    df: pd.DataFrame, encoders: dict[str, LabelEncoder], feature_names: list[str]
) -> pd.DataFrame:
    """学習時に保存したLabelEncoderを使って推論用データを変換する
    （新規fitはしない。未知カテゴリは-1扱い）。"""
    df = df.drop(columns=[c for c in ID_COLS + CONSTANT_COLS + [TARGET_COL] if c in df.columns])

    X = df.copy()
    for col, le in encoders.items():
        if col in X.columns:
            known = set(le.classes_)
            X[col] = X[col].map(lambda v, le=le, known=known: le.transform([v])[0] if v in known else -1)

    return X[feature_names]


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> tuple[lgb.LGBMClassifier, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "report": classification_report(y_test, y_pred, target_names=["No", "Yes"]),
        "y_test": y_test,
        "y_proba": y_proba,
    }
    return model, metrics


def evaluate_thresholds(
    y_true: pd.Series, y_proba, thresholds: list[float] | None = None
) -> pd.DataFrame:
    """複数の判定閾値に対するPrecision/Recall/F1を一覧化する。

    仕様書の方針「離職の予兆を見逃さない」を優先するため、閾値を0.5より
    下げてRecallの変化を確認する用途。
    """
    if thresholds is None:
        thresholds = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]

    rows = []
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        rows.append(
            {
                "threshold": t,
                "precision": precision_score(y_true, y_pred_t, zero_division=0),
                "recall": recall_score(y_true, y_pred_t, zero_division=0),
                "f1": f1_score(y_true, y_pred_t, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def classify_with_threshold(proba: float, threshold: float = DEFAULT_THRESHOLD) -> int:
    return int(proba >= threshold)


def get_feature_importance(model: lgb.LGBMClassifier, X: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return importance.head(top_n).reset_index(drop=True)


def score_employee(model: lgb.LGBMClassifier, X_row: pd.DataFrame) -> float:
    """1名分の特徴量(1行DataFrame)から離職リスク確率を返す。"""
    return float(model.predict_proba(X_row)[:, 1][0])


def top_risk_factors_for_employee(
    model: lgb.LGBMClassifier, X_row: pd.DataFrame, top_n: int = 5
) -> pd.DataFrame:
    """モデル全体のfeature importance上位のうち、その従業員の値も併記して返す
    （SHAP等は使わず、仕様書どおりfeature_importances_ベースの簡易表示）。

    日給・月収・月額レート・時給はいずれも給与水準を表す指標で、並べて表示すると
    冗長で分かりにくいため「月収」に一本化する（重要度は給与関連の最大値を採用）。"""
    full = get_feature_importance(model, X_row, top_n=len(X_row.columns))

    salary_mask = full["feature"].isin(SALARY_FEATURES)
    if salary_mask.sum() > 1:
        salary_importance = float(full.loc[salary_mask, "importance"].max())
        full = full.loc[~salary_mask]
        rep = pd.DataFrame([{"feature": "MonthlyIncome", "importance": salary_importance}])
        full = pd.concat([full, rep], ignore_index=True)
        full = full.sort_values("importance", ascending=False)

    importance = full.head(top_n).reset_index(drop=True)
    importance["employee_value"] = importance["feature"].apply(lambda f: X_row.iloc[0][f])
    return importance


# 給与関連は「低い」ほど危険
SALARY_FEATURES = {"MonthlyIncome", "DailyRate", "HourlyRate", "MonthlyRate"}
# 満足度系は「低い」ほど危険
SATISFACTION_FEATURES = {"JobSatisfaction", "WorkLifeBalance", "EnvironmentSatisfaction", "RelationshipSatisfaction"}
INVERSE_RISK_FEATURES = SALARY_FEATURES | SATISFACTION_FEATURES


def get_risk_direction(feature: str) -> int:
    """値が大きいほど危険なら+1、小さいほど危険なら-1を返す。

    平均との差(%)にこの符号を掛けると、プラスなら常に「離職リスクを高める方向」、
    マイナスなら常に「定着に寄与する方向」という向きに統一できる
    （例：給与が平均より低い＝pct_diffは負だが、危険方向としては+扱いにしたい）。
    """
    return -1 if feature in INVERSE_RISK_FEATURES else 1


def get_factor_risk_color(feature: str, employee_val: float, dataset_df: pd.DataFrame) -> str:
    """従業員の値が全体平均と比べてどのくらい危険かを判定し、色を返す。
    赤(high)・黄(mid)・緑(low)。"""
    if feature not in dataset_df.columns:
        return "low"

    try:
        avg = dataset_df[feature].mean()
        if avg == 0:
            return "low"

        pct_diff = ((employee_val - avg) / abs(avg)) * 100
        risk_impact = pct_diff * get_risk_direction(feature)

        if risk_impact > 30:
            return "high"
        elif risk_impact > 10:
            return "mid"
        return "low"
    except (TypeError, ValueError, ZeroDivisionError):
        return "low"


def save_model(
    model: lgb.LGBMClassifier,
    encoders: dict[str, LabelEncoder],
    feature_names: list[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "lgbm_model.pkl")
    joblib.dump(encoders, MODEL_DIR / "encoders.pkl")
    joblib.dump(feature_names, MODEL_DIR / "feature_names.pkl")
    joblib.dump(threshold, MODEL_DIR / "threshold.pkl")


def load_model() -> tuple[lgb.LGBMClassifier, dict[str, LabelEncoder], list[str], float]:
    model = joblib.load(MODEL_DIR / "lgbm_model.pkl")
    encoders = joblib.load(MODEL_DIR / "encoders.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    threshold_path = MODEL_DIR / "threshold.pkl"
    threshold = joblib.load(threshold_path) if threshold_path.exists() else DEFAULT_THRESHOLD
    return model, encoders, feature_names, threshold
