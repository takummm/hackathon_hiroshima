"""学習・評価・feature importance確認用のCLIスクリプト（8/21タスク）。

実行:
    venv/Scripts/python.exe src/train_model.py
"""

from model import (
    DEFAULT_THRESHOLD,
    evaluate_thresholds,
    get_feature_importance,
    load_data,
    preprocess,
    save_model,
    train_and_evaluate,
)


def main() -> None:
    df = load_data()
    X, y, encoders = preprocess(df)

    model, metrics = train_and_evaluate(X, y)

    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"ROC AUC : {metrics['roc_auc']:.3f}")
    print()
    print(metrics["report"])

    print("=== Feature Importance Top10 ===")
    print(get_feature_importance(model, X, top_n=10).to_string(index=False))

    print()
    print("=== 閾値ごとのPrecision/Recall/F1（見逃さない方針での調整） ===")
    threshold_table = evaluate_thresholds(metrics["y_test"], metrics["y_proba"])
    print(threshold_table.to_string(index=False))

    chosen = threshold_table.loc[threshold_table["threshold"] == DEFAULT_THRESHOLD].iloc[0]
    print(
        f"\n採用閾値: {DEFAULT_THRESHOLD} "
        f"(Precision={chosen['precision']:.3f}, Recall={chosen['recall']:.3f}, F1={chosen['f1']:.3f})"
    )

    save_model(model, encoders, list(X.columns), threshold=DEFAULT_THRESHOLD)
    print("モデルと閾値を models/ に保存しました。")


if __name__ == "__main__":
    main()
