"""アプリ全体で共有するリスク配色定数。

信号機カード（🔴🟡🟢バッジ・メトリクス）、原因分析のダイバージング・バーチャート、
リスクスコア表示（閾値0.30の判定色）など、リスクを色で示す箇所は全てここを参照し、
表記ゆれを防ぐ。
"""

RISK_HIGH = "#E63946"
RISK_MID = "#F4A261"
RISK_LOW = "#2A9D8F"

RISK_TIER_COLOR: dict[str, str] = {"high": RISK_HIGH, "mid": RISK_MID, "low": RISK_LOW}
