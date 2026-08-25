# 仕様書（Claude Code向け・技術仕様）v0.5

対象プロジェクト: 広島県企業向け AI離職予防・人材定着支援サービス
最終更新: 2026-08-25（v0.4からの変更: 新規タスクの扱いを確定、コードフリーズ方針を追加、スケジュール更新）

---

## ⚠️ v0.5での重要な決定事項

- **UI改善フェーズは終了。** Claude Designでの改善案も検討したが効果は限定的と判断し、以降はUIに手を入れない。
- **8/26以降、コード変更は原則フリーズする。** 目的はプレゼン資料・台本作成に集中するため。許容されるのは以下のみ：
  - デモ実行時に発覚した**バグ修正**（高・中・低リスク3パターンのデモが壊れる場合は最優先で対応）
  - プレゼン用のグラフ・数値の**書き出し**（既存機能からのエクスポート程度。新規ロジック追加は避ける）
- **TA提案（総合プロフィール分析）は本番実装を見送り、`docs/future_scope.md` に構想メモとして記録する。** プレゼンでは「将来構想」パートで言及するのみとし、コード実装（`analyze_employee_profile()`等）は着手しない。理由：残りスケジュールがプレゼン準備一本のため、新機能追加の時間対効果が低いと判断。

---

## 実装済み（変更なし・現状維持）

以下はすべて実装・動作確認済み。**新規タスクではリグレッションに注意すること。**

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
├── design/
│   └── dashboard_reference.html  # Claude Designで作成した参考デザイン（不採用・アーカイブ扱い）
└── docs/
    └── future_scope.md           # TA提案（総合プロフィール分析）の構想メモ（新規・v0.5で追加予定）
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

## 凍結タスク（TA提案・実装見送り確定）

TAから「離職リスクだけでなく、従業員を丸ごと測定できるといいのでは」という提案があった。
**v0.5時点で、本番実装は見送ることを確定した。** 以下は将来構想としての記録であり、当面着手しない。

### 提案内容（記録用）
現状は離職リスクという単一指標のみを算出している。将来的にこれを拡張し、従業員の以下の側面も可視化する構想：

1. **成長曲線**（Growth Trajectory）
   - 勤続年数・昇進履歴（`YearsAtCompany`, `YearsSinceLastPromotion`等の既存列を活用）から、成長ペースを可視化
   - 実装案: 既存データ列から簡易スコアを算出し、折れ線グラフ or ゲージチャートで表示

2. **強み・弱み**（Strengths / Weaknesses）
   - Feature importanceの発想を応用し、個人の特徴量のうち「平均より良い/悪い」項目を抽出
   - 実装案: 従業員個人の値と全体平均を比較し、上位/下位の特徴量をLLMに渡して自然文化

3. **性質・特質**（Traits）
   - `JobSatisfaction`, `WorkLifeBalance`, `EnvironmentSatisfaction`等の満足度系カラムを組み合わせ、傾向をLLMに要約させる

### 実装方針（着手する場合の設計案・参考記録）
- 既存のMVP機能は変更しない（審査直前のリグレッションリスクを避けるため）
- 新機能は既存の「原因分析」「施策提案」と並列に、追加タブ or 追加セクションとして実装
- 新規関数案: `src/llm.py` に `analyze_employee_profile()` を追加し、成長曲線・強み弱み・性質の3点を1回のLLM呼び出しでまとめて生成（API呼び出し回数を増やさない設計）
- UIには「総合プロフィール分析」ボタンを追加し、クリックで実行（既存の2ボタンと並べる）

### 確定した対応
- **コード実装は行わない。**
- `docs/future_scope.md` に上記の構想メモを転記し、プレゼンの「将来構想」スライドの参考資料とする。
- プレゼンでの言及は口頭・スライド上の1トピックにとどめ、デモでは見せない。

---

## 残りスケジュール（v0.5更新・プレゼン準備中心）

| 日付 | フェーズ | Claude Codeへの依頼事項 |
|------|---------|------------------------|
| 8/26 | 課題スライド作成 | 基本的にコード作業なし。もし広島統計データのグラフ化スクリプトが必要な場合のみ小規模対応 |
| 8/27-28 | プレゼン資料最終化・台本作成 | コード変更なし想定。`docs/future_scope.md` 作成のみ対応可 |
| 8/29 | リハーサル | デモ動作確認（バグがあれば最小修正で対応） |
| 8/30 | 予備日 | 直前バグ修正のバッファ |
| 8/31 | 提出 | — |

---

## 注意事項（Claude Codeへの申し送り）
- ユーザーはPython中心の学習歴で、フロントエンド（HTML/CSS/JavaScript）は基礎理解のみ。技術説明は平易に。
- Streamlit の制約（HTML/CSS直接編集不可、rerunで上書きされる）は理解済みなので、再度の説明は不要。
- **8/26以降はコードフリーズが基本方針。** 新機能の提案・実装依頼があっても、まず「本当にこの時間対効果でやるべきか」をユーザーに確認してから着手すること。
- デモの安定性（高・中・低リスク3パターンが正常動作すること）が最優先。これを壊すリスクのある変更は行わない。
