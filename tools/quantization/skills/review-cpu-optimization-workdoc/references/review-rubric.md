# CPU最適化workdoc監査rubric

## 判定

- PASS: 別の作業者が迷わず実行でき、対象の主張が必要証拠に支えられる。
- PASS_WITH_NOTES: 非阻害の小さな改善点のみ残る。
- REVISE: 目的、実行条件、gate、測定、公平性、証拠に実行を妨げる欠落がある。
- FAIL: 要求された機能を省略、結果を捏造、または重要な精度/安全条件に矛盾する。

計画の判定と実装完了の判定を分ける。未完成実装を計画レビューPASSで代替しない。

## Blockerを疑う項目

- 固定したbaseline/candidateを同じ入力と設定で再実行できない。
- model/prompt/frame/metricが不明、または候補結果を見てから閾値を変更した。
- GGUF読込をmodel実行、CPU浮動小数点復元をINT8kernelと主張している。
- operator/image成功をfullvideo、stateful tracking、最終UI成功へ拡大している。
- native graph未接続、必要演算未対応、fallbackなしで完了と主張する。
- private datasetやweightsを公開する権限がないのに公開対象に含む。
- required artifactがなく、過去runやsyntheticを今回の実モデル結果として扱う。

- cache keyにmodel/quant/preprocess/image/policy/conditioning依存が欠け、他objectのstateや別モデルのfeatureを再利用する。

## Majorを疑う項目

- profileなしにbottleneckを断定、比較時のthread/build/hardwareが異なる。
- paired反復がない、best-caseだけ採用、共有負荷が未記録、conversion費用が隠れる。
- memoryのcold/warm・process scope・単位が不明。
- shape/stride/alignment/lifetime、unsupported ISA、ゼロ/empty/異常入力が未検証。
- 再量子化やattention等の誤差がkernelだけで検証され、model gateが未再実行。
- 候補の採用がmicrobenchだけで、代表end-to-end効果や統合回帰を確認していない。
- copy/adaptationの出典・ライセンス・noticeが確認されていない。
- checklistが複数操作を一項目にまとめ、即時記録や失敗時再開条件がない。

- cold/warm/resume同値、RAM/disk上限、破損cache拒否、prompt変更後suffix再計算が未検証。

## Coverage checklist

各項目を adequate / partial / missing で判定し、証拠へリンクする。

- ゴール・必須機能・非ゴール・traceability
- repo/環境/入力/protocolの固定と実行コマンド
- profileと候補選定の根拠
- 正確な数値契約、oracle、固定model-level gates
- paired timing、全samples、conversion/setup、peak memory
- portable fallback・thread・layout・lifetime・異常系
- 最小interface、既存backend再利用、SOLID/KISS/DRY
- graph/application統合と回帰
- atomic checklist、実日時、即時記録、DoD
- private/public分離、license/provenance、公開権限

## 証拠の扱い

「未実行」「実行中」「失敗」「測定不足」と「成功」を別にする。報告文の数字を
raw reportと照合し、artifact/hash、codeversion、actualcommandを確認する。
結果JSONが後生成でも事前の閾値承認記録があれば区別して記載できる。

未知のhardwareへの速度保証、疎なsampleから全datasetへの精度保証、確認済みIDと
意味的同一性の混同、未確認のGT化を避ける。正しい非採用結論も認める。
