# 使用したAI技術一覧

## 機械学習モデル

### LightGBM（勾配ブースティング決定木）
- **用途**: 従業員ごとの離職リスクスコア算出
- **学習データ**: IBM HR Analytics Employee Attrition（Kaggle公開データセット、従業員1,470名分）
- **設定**: `class_weight="balanced"`（離職者が少数派のため重み補正）
- **評価指標**: Accuracy 0.840 / ROC-AUC 0.798 / Recall（離職クラス）0.49
- **判定閾値**: 0.30（「離職の予兆を見逃さない」方針のため、通常の0.5より意図的に低く設定）
- **主要な特徴量重要度**: MonthlyIncome, DailyRate, Age, DistanceFromHome, MonthlyRate

## 生成AI（LLM）

### Claude API（Anthropic）
- **開発時のモデル**: Claude Haiku 4.5（低コスト・高速）
- **デモ本番時のモデル**: Claude Sonnet 5（品質優先）※UIのトグルで切替可能
- **用途1: 原因分析** — リスクスコアと主要リスク要因（feature importance上位）をもとに、離職リスクの判定理由を自然文で説明
- **用途2: 定着施策の提案（簡易RAG）** — 自社の制度一覧（`data/company_policies.txt`）を検索対象データとして与え、原因分析の結果に対応する具体的な制度を引用しながら定着施策を提案

## その他の技術スタック

| 技術 | 用途 |
|---|---|
| Streamlit | Webアプリケーションフレームワーク（UI・状態管理） |
| Altair | データ可視化（部署別リスクヒートマップ、重要度グラフ） |
| pandas | データ処理・集計 |
| scikit-learn | モデル評価（train_test_split、各種指標） |
| python-dotenv | APIキー等の環境変数管理 |
