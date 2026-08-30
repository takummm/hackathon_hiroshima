# 開発引き継ぎメモ（新チャット用コンテキスト）

最終更新: 2026-08-31。コンテキストウィンドウ節約のため会話を分割。次のチャットの冒頭でこのファイルを読み込めば、これまでの経緯を把握できる。

## プロジェクト概要

- **サービス名**: AI離職予防・人材定着支援サービス（広島県企業向け、ハッカソンMVP）
- **提出期限**: 2026-08-31（**本日が提出日**）
- **場所**: `C:\Users\taku3\OneDrive\ドキュメント\hackathon_project`（独立したgitリポジトリ、`master`ブランチ）
- **仕様書**: `SPEC_FOR_CLAUDE_CODE.md` は**ユーザーがリポジトリから削除済み**（コミット 29d5704）。以前は v0.7。参照が必要な場合はユーザーに確認すること。
- **本番プレゼン**: 発表7分＋質疑8分＝合計15分

## 技術スタック

- Python（LightGBM, Streamlit 1.62, Claude API(`anthropic` 0.125), pandas, Altair, scikit-learn, python-dotenv）
- モデル: LightGBM（`class_weight="balanced"`、判定閾値0.30）、IBM HR Analytics Employee Attritionデータで学習
- LLM: Claude API（開発時 Haiku 4.5 `claude-haiku-4-5-20251001` / デモ本番 Sonnet 5 `claude-sonnet-5`、サイドバーのトグルで切替）
- 使用AI技術一覧: `submission/AI活用箇所一覧.md`（`docs/ai_technologies.md` と `submission/ai_technologies.md` は削除済み）

## ファイル構成（主要なもの）

```
hackathon_project/
├── app.py                        # Streamlitメインアプリ（3タブ構成）
├── src/
│   ├── model.py                  # モデル読込・スコア算出・要因抽出
│   │                             #   - top_risk_factors_for_employee(): 給与系4カラムを「月収」に集約
│   │                             #   - risk_contributions_for_employee(): ★個人単位の寄与度（後述）
│   ├── llm.py                    # explain_risk_factors(), suggest_interventions()
│   │                             #   - _response_text(): ThinkingBlock対策
│   │                             #   - format_risk_pct(): 「0.1%未満」表記
│   │                             #   - _THINKING_DISABLED: thinking無効化
│   ├── labels.py                 # 日本語ラベル変換（列名・カテゴリ値・順序尺度・円換算）
│   └── colors.py                 # RISK_HIGH/MID/LOW配色の一元管理（変更不可の共有パレット）
├── data/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv
│   └── company_policies.txt      # 簡易RAG用
├── models/                       # 学習済みLightGBMモデル一式
├── .streamlit/config.toml        # ライトテーマ設定（サーバー起動時のみ読込）
├── docs/
│   ├── ai_technologies.md        # ※削除済み
│   ├── future_scope.md           # 将来構想メモ
│   └── dev_handoff.md            # このファイル
└── submission/                   # 提出物集約フォルダ
    ├── README.md
    ├── service_overview.md
    ├── AI活用箇所一覧.md          # 未コミット（下記「git状態」参照）
    └── _提出チェックリスト.md
```

## アプリのUI構成（タブ3構成）※2026-08-30〜31に大幅改修

### タブ1: 全体リスク一覧 & ターゲット選定
1. 「元データ（CSV）を表示」展開ボックス
2. **「🔒 スマートリング入力（今後対応予定・現在未接続）」展開ボックス**（cccf674で追加）
   - **完全に非機能の静的表示のみ**。説明文と `st.caption` のみ。ボタン・アップローダー・接続処理は一切なし。モデル入力・AI呼び出しに関与しない。
3. **「離職リスクの高い部門・職種」ランキング**（df1ee23で旧ヒートマップを廃止して刷新）
   - 「特に注意が必要な部門・職種」= 上位3件の `st.metric` カード
   - 横棒グラフ（Altair）: 部門×職種の平均予測リスク降順、11組、1位のみネイビー(#1e3a8a)強調・他は淡いブルー(#bfdbfe)
   - 各バー右に「X%（N人）　区分」を直接表示。凡例なし・軸目盛りなし
   - 区分ラベルは `RISK_LEVEL_LABEL`（llm.get_risk_tier準拠: 70%↑=高リスク / 10%↓=低リスク / それ以外=要注意）
   - **バーをクリックすると `st.altair_chart(on_select="rerun")` + `alt.selection_point(name="pick")` で下の「組織全体のリスク一覧」がその部門・職種に絞り込まれる**。同じバー再クリックで解除。session_stateは使わずチャートの選択状態を利用
4. 「組織全体のリスク一覧」テーブル（絞り込み対応、絞り込み時はサマリー指標も連動）
5. 「ターゲット選定」: デモ用ショートカット（高#622／中#1203／低#1804）、EmployeeNumber selectbox

### タブ2: 個別詳細カルテ
- リスクスコア表示: **微小非ゼロ値は「0.1%未満」表記**（`risk_score_text`、app.py 360行付近）
- 判定バッジ: 「⚠️ 要注意」/ **「問題なし」**（旧「OK」から変更）
- 左: 従業員プロフィール表
- 右: **「離職リスクへの影響」ダイバージング横棒チャート**（87b94e0で「重要度グラフ」を刷新）
  - `risk_contributions_for_employee()` = **LightGBM組み込みTreeSHAP** `model.booster_.predict(X, pred_contrib=True)` による個人単位の寄与度（log-odds空間、符号つき）。**`shap`パッケージは不使用・未インストール**
  - 影響度は最大=100の相対スケールに正規化して表示（生のlog-odds値は見せない）
  - リスクを高める要因＝ネイビー(#1e3a8a)、下げる＝淡いブルー(#93c5fd)。**赤系は使わない**
  - 全要因が同方向のときは凡例なし・単色、方向混在時のみ2色＋凡例
  - 軸タイトルは「離職リスクへの影響度」のみ
- 「本人と全社平均の比較」テーブル（要因／この従業員の値／全社平均／平均との差の4列。「影響」列は削除済み）

### タブ3: AIによる提案
- 原因分析（`explain_risk_factors`）、定着施策の提案（`suggest_interventions`、簡易RAG）

## src/llm.py の重要な実装（2026-08-30に3件のバグ修正）

- **`_response_text(response)`**: Sonnet 5 は `content` 先頭に `ThinkingBlock`（`.text` なし）を返すことがあり `content[0].text` が `AttributeError`。type=="text" のブロックのみ結合して返す。
- **`thinking=_THINKING_DISABLED`（`{"type": "disabled"}`）**: 思考トークンが `max_tokens` を圧迫して生成が途中で切れるのを防止。両API呼び出しに付与。
- **`max_tokens`**: explain 400→700 / suggest 500→900。
- **`format_risk_pct(risk_pct)`**: `0 < risk_pct < 0.1`（％単位）で「0.1%未満」。両プロンプトの「# 離職リスクスコア」に適用（LLMに「リスクゼロ」と誤解させない）。

## 運用上の注意点（ハマりやすいポイント）

- **`venv\Scripts\streamlit.exe` を直接実行するとWindowsのポリシーにブロックされる。** 必ず `venv/Scripts/python.exe -m streamlit run app.py` の形式。
- ローカル検証はポート8502以降で行い、該当PIDのみ狙って停止（8501はユーザーが使用中のことが多い）。
- Bashでのシェルリダイレクトは日本語が文字化け（cp932）。Pythonスクリプト内で `encoding='utf-8'` 指定してファイル書き出し→Readが確実。コンソール出力の文字化けは無視してよい。
- `st.tabs` はJSで `[role="tab"]` 要素を `.click()` する方が確実。
- `.streamlit/config.toml`（テーマ）はサーバー起動時のみ読込。テーマ変更時は完全再起動が必要。
- Browser paneの `computer{action:"screenshot"}` はタイムアウトすることがある→リトライで通る。`read_page`/`javascript_tool` でDOM確認も併用。
- **コミット確認は毎回不要**（軽微な変更は直接コミット）。仕様変更・機能追加など重要なもののみ確認（[[feedback-commit-confirmation]] メモリ参照）。
- **コードフリーズ方針**（[[project-hackathon-code-freeze]]）: 提出期限8/31。コード変更は原則、仕様書優先リスト＋デモが壊れるバグ修正に限定。ただし今セッションはユーザー主導のUI改善依頼が多数あり、都度対応した。

## デモ本番の3パターン（リグレッション確認に使用）

| ラベル | EmployeeNumber | スコア | 判定 |
|---|---|---|---|
| 高リスク | #622 | 99.2% | ⚠️ 要注意 |
| 中リスク | #1203 | 49.6% | ⚠️ 要注意 |
| 低リスク | #1804 | 0.1%未満 | 問題なし |

※ #1804 の生スコアは約0.04%（全従業員中の最小値）。丸めで0.0%に見えるのを「0.1%未満」表記で回避している（バグではない）。

## このセッション（2026-08-30〜31）のコミット履歴

```
0541de0 タイトル下の説明文を1行表示に変更（max-width撤廃・nowrap）
29d5704 提出物構成の整理: ルートのREADME/仕様書を削除し submission/ に集約
cccf674 タブ1にスマートリング入力（今後対応予定）のモックアップ表示を追加
71354b8 LLMプロンプトに渡すリスクスコアも「0.1%未満」表記に統一
5e225ad 施策提案・原因分析の生成が途中で切れる問題を修正（thinking無効化・max_tokens増）
89361e5 Sonnet 5利用時にLLM応答取得でAttributeErrorが出る問題を修正（_response_text）
df1ee23 部門・職種別リスクをヒートマップからランキングUIに刷新
75dd602 離職リスクへの影響グラフを赤系から青系の控えめな配色に変更
34c7279 「主要リスク要因」画面の見せ方を簡素化（凡例削減・相対スケール化・表の役割分離）
87b94e0 主要リスク要因グラフを「離職リスクへの影響」ダイバージングチャートに刷新（TreeSHAP採用）
acad289 判定表示の「OK」を「問題なし」に変更
90ab45e 離職リスクスコアが微小値のとき「0.0%」と誤表示される問題を修正
92093a1 重要度グラフとヒートマップの視認性を改善（※後続コミットで大幅に置き換え）
```

## 現在のgit状態（未コミット）

```
?? Anthropic_slide/                    ← 中に .git がある（ネストしたgitリポジトリ）。コミットすると壊れたgitlinkになるので add しないこと
?? demo_vol2.mp4                        ← 約31MBの動画。コミット是非はユーザー確認
?? slide/                              ← slide_v2〜v6.pptx + "by claude design.pptx"。どれが最新かユーザー確認
?? submission/AI活用箇所一覧.md          ← 提出物#5の実体（コミット検討）
```

前回のチャットで `git add -u` により tracked な削除・変更（README.md・SPEC_FOR_CLAUDE_CODE.md・submission/ai_technologies.md・スライド案/hackathon_pitch_v1.pptx の削除、submission/README.md の編集）はコミット済み（29d5704）。

## Claudeメモリに保存済みの情報（セッション跨ぎで自動読込）

- **コミット確認方針**: 軽微な変更は確認なしでコミット、重要なもののみ確認
- **コードフリーズ方針**: 提出期限8/31、TAミーティング8/27。コード変更は仕様書の優先リスト＋デモ破壊バグに限定

## 未完了・今後の対応候補

- デモ動画（3分以内）の撮影・編集（`demo_vol2.mp4` が候補か？ 要確認）
- プレゼン資料（PDF書き出し）の `submission/` への配置。スライドは `slide/` に v2〜v6 が並ぶ
- `submission/AI活用箇所一覧.md` のコミット、`_提出チェックリスト.md` との整合確認
- 提出物一式（PDF・デモ動画・README・サービス概要・AI技術一覧）の最終確認
