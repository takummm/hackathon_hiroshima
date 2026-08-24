# 仕様書（Claude Code向け・技術仕様）v0.4

対象プロジェクト: 広島県企業向け AI離職予防・人材定着支援サービス
最終更新: 2026-08-25

---

## 実装済み（変更なし・現状維持）

以下はすべて実装・動作確認済み。新規タスクではリグレッションに注意すること。

### 技術スタック
- 言語: Python（全体）
- MLモデル: LightGBM（`class_weight="balanced"`、判定閾値 0.30）
- UI: Streamlit（フロント・バック共にPython、分離なし）
- LLM: Claude API（開発時 Haiku 4.5 / デモ本番 Sonnet 5、UIトグルで切替）
- データセット: IBM HR Analytics Employee Attrition（Kaggle公開データ）

### モデル評価指標
- Accuracy: 0.840 / ROC-AUC: 0.798
- Recall（離職クラス）: 0.49（class_weight="balanced"適用後、閾値0.30で運用）
- Feature importance上位: MonthlyIncome, DailyRate, Age, DistanceFromHome, MonthlyRate

### ファイル構成
```
hackathon_project/
├── app.py                        # Streamlitメインアプリ
├── src/
│   ├── model.py                  # transform_with_encoders(), validate_dataset(), friendly_data_error_message()
│   └── llm.py                    # explain_risk_factors(), suggest_interventions()
├── data/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv
│   └── company_policies.txt      # 簡易RAG用（6項目）
├── models/
│   ├── lgbm_model.pkl
│   ├── encoders.pkl
│   └── feature_names.pkl
└── design/
    └── dashboard_reference.html  # Claude Designで作成した参考デザイン（未反映分あり）
```

### 実装済み機能（MVP 5ステップ）
1. CSVアップロード・従業員選択（デモ用ショートカット: 高#622/中#1203/低#1804）
2. 離職リスクスコア表示（%、閾値0.30に対する判定）
3. Feature importance上位5件表示
4. `explain_risk_factors()`：原因分析（自然文）
5. `suggest_interventions()`：簡易RAGベースの施策提案

### エラーハンドリング（8/23実装済み）
- API認証失敗・タイムアウト → `friendly_error_message()` でユーザー向けメッセージに変換
- CSV異常データ（欠損値・列欠落・型混入） → `validate_dataset()` + `friendly_data_error_message()` で保護
- `st.cache_resource`（モデル）/ `st.cache_data`（データ・LLM結果）でrerun時の重い処理再実行を回避
- 従業員ID別にsession_stateを管理し、切替時に古い分析結果が残らないようにしている

---

## 新規タスク（TA提案・要検討）

TAから「離職リスクだけでなく、従業員を丸ごと測定できるといいのでは」という提案があった。
以下の拡張を **時間の許す範囲で** 検討する。優先度は低〜中（プレゼンの完成度を優先）。

### 提案内容
現状は離職リスクという単一指標のみを算出している。これを拡張し、従業員の以下の側面も可視化する：

1. **成長曲線**（Growth Trajectory）
   - 勤続年数・昇進履歴（`YearsAtCompany`, `YearsSinceLastPromotion`等の既存列を活用）から、成長ペースを可視化
   - 実装案: 既存データ列から簡易スコアを算出し、折れ線グラフ or ゲージチャートで表示

2. **強み・弱み**（Strengths / Weaknesses）
   - Feature importanceの発想を応用し、個人の特徴量のうち「平均より良い/悪い」項目を抽出
   - 実装案: 従業員個人の値と全体平均を比較し、上位/下位の特徴量をLLMに渡して自然文化

3. **性質・特質**（Traits）
   - `JobSatisfaction`, `WorkLifeBalance`, `EnvironmentSatisfaction`等の満足度系カラムを組み合わせ、傾向をLLMに要約させる

### 実装方針（提案）
- **既存のMVP機能は変更しない**（審査直前のリグレッションリスクを避けるため）
- 新機能は既存の「原因分析」「施策提案」と並列に、**追加タブ or 追加セクション**として実装する
- 新規関数案: `src/llm.py` に `analyze_employee_profile()` を追加し、成長曲線・強み弱み・性質の3点を1回のLLM呼び出しでまとめて生成する（API呼び出し回数を増やさない設計にする）
- UIには「総合プロフィール分析」ボタンを追加し、クリックで実行（既存の2ボタンと並べる）

### 実装しない場合の代替案
時間が足りない場合は、**プレゼンの「将来構想」パートで言及するのみ**とし、コード実装は見送る。
その場合、`docs/future_scope.md`のようなファイルに構想メモとして残しておくことを推奨。

### 判断基準
- 8/26時点で実装に着手できるかを判断する
- 着手する場合も、**既存デモ（高・中・低リスク3パターン）が壊れないことを最優先**で確認しながら進める
- 新機能のためにMVPの安定性を犠牲にしない

---

## 注意事項（Claude Codeへの申し送り）
- ユーザーはPython中心の学習歴で、フロントエンド（HTML/CSS/JavaScript）は基礎理解のみ。技術説明は平易に。
- Streamlit の制約（HTML/CSS直接編集不可、rerunで上書きされる）は理解済みなので、再度の説明は不要。
- 残りスケジュールがタイトなため（8/26以降はスライド作成・リハーサルが本題）、新機能実装は**時間対効果を都度確認しながら**進めること。
