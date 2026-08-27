# AI離職予防・人材定着支援サービス

広島県企業向け、AIを活用した従業員の離職リスク予測・原因分析・定着施策提案サービス（ハッカソンMVP）。

## できること

1. 組織全体の離職リスクを一覧・ヒートマップで俯瞰
2. 従業員個人の離職リスクスコアをAIモデルで算出
3. 主要リスク要因を平均値との比較付きで表示
4. Claude APIによる原因分析（自然文）
5. 自社の制度一覧を踏まえた定着施策の提案（簡易RAG）

## 動作環境

- Python 3.10以上（開発環境: 3.14.4）
- Windows / Mac / Linux いずれでも動作可能

## セットアップ

```bash
pip install -r requirements.txt
```

### Claude APIキーの設定

プロジェクトルートに `.env` ファイルを作成し、以下を記載してください。

```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
```

`.env` ファイルは第三者と共有しないでください。

## アプリの起動

```bash
streamlit run app.py
```

起動後、自動的にブラウザで `http://localhost:8501` が開きます。開かない場合は手動でアクセスしてください。

## デモの使い方

1. 「全体リスク一覧 & ターゲット選定」タブで、デモ用ショートカットボタンから従業員を選択
   - 高リスクの例（#622／約99%）
   - 中リスクの例（#1203／約50%）
   - 低リスクの例（#1804／約0%）
2. 「個別詳細カルテ」タブで、リスクスコア・従業員プロフィール・主要リスク要因を確認
3. 「AIによる提案」タブで「原因分析を実行」→「施策提案を実行」の順にボタンを押す

## プロジェクト構成

```
hackathon_project/
├── app.py                        # Streamlitメインアプリ
├── src/
│   ├── model.py                  # モデル読込・スコア算出・要因抽出
│   ├── llm.py                    # 原因分析・施策提案（Claude API）
│   ├── labels.py                 # 列名・カテゴリ値の日本語表示変換
│   └── colors.py                 # リスク配色の一元管理
├── data/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv  # 学習・分析用データ
│   └── company_policies.txt      # 簡易RAG用の自社制度一覧
├── models/                       # 学習済みモデル（LightGBM）
└── docs/
    └── ai_technologies.md        # 使用したAI技術一覧
```

## 使用技術

主要な技術スタックの一覧は [docs/ai_technologies.md](docs/ai_technologies.md) を参照してください。
