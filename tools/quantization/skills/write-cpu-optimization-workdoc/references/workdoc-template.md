# CPU最適化 作業計画書兼記録書の構造

このファイルは構成の指針。出力文書では各項目を実在するrepo情報、確定した
要求、具体的なコマンドと証拠に置き換える。空の雛形を納品しない。

## 1. 作業目的

直下に「ゴール要求分析」を置く。

- ユーザーの直感的な目的、必須機能、精度・速度・memoryの意味
- 明示要求、暗黙制約、非ゴール、未解決の判断
- TR-IDごとのsubgoal、成果物、テスト、DoDの対応
- 既存モデル・品質gateの維持、事後閾値変更の禁止

## 2. 実行条件と調査

- 実日時・timezone、repo/branch/HEAD、既存差分、編集所有者
- hardware/ISA/CPU topology、OS、compiler、backend commit、build flags
- uv/just/CMake等の標準経路と正確なworkdir
- model/config/input/fixture hashes、prompt/frame/seed/thread設定
- 実profile、bottleneck、読み取りsource、license/provenance
- storage/reference/kernel/graph/applicationを分けた現状表

## 3. 設計と事前登録

- 最適化候補と最小interface、ownership、buffer lifetime
- fallback条件、unsupported shape/ISA、layoutとstride
- 数値誤差源とoracle、既存model-level gate
- benchmark shapes、warmup、paired AB/BA回数、固定thread
- conversion/setup/inference/end-to-end時間の切り分け
- peak memoryの単位、process scope、cold/warm状態
- 採否基準、noiseや未実行の扱い、scope変更時の停止条件
- 該当時: encoder共有境界、完全cache key、objectstate隔離、RAM/disk予算
- 該当時: cache export/import/resume、破損/失効、prompt編集後suffix再計算

## 4. 原子的チェックリスト

調査→設計/レビュー→失敗試験→実装→検証→統合→納品を分ける。
各「手順」はSKILL.mdの四行形式に従い、一操作だけを記述する。
既存の承認された作業書がある場合はその順序と番号を維持する。

## 5. 精度・性能・memory検証

- baselineとcandidateのraw outputs、差分、全casesの成否
- 複数実行の全samples、median/分布、measurement noise
- 同じgraph/入力/資源による比較、conversionを含む利用時の費用
- portable fallback、異なるthread、異常入力の回帰
- model reload、fullgraph品質、application要求の再確認
- operatorだけの成果とend-to-end成果を別行にする

## 6. 完了の定義

各TR-IDに独立したDoD checkboxを置く。実装・品質・性能判断・fallback・
統合・再現性・license/公開監査を証拠で判定する。候補の不採用も理由と
測定が十分なら最適化調査の正しい結論だが、残るユーザー機能は省略しない。

## 7. 作業記録

実日時、手順/TR、コマンドまたは編集、結果、artifact/hash、次の条件を記す。

### 注意事項

- 実行前にdateで時刻を取得する。撮影日時や推測の経過時間と混同しない。
- 一項目だけ実行・検証し、チェックした直後に結果を記録する。
- エラーは隠さず、項目を未完了のまま再開条件を残す。
- DRY/KISS/SOLIDをrepoの設計方針に沿って適用する。
- synthetic、reference、実モデル、native、最終UIの証拠を混同しない。
- private入力・weights・raw logsを無断公開しない。必要なlicenseを保持する。

## 8. レビューと残課題

PASS / PASS_WITH_NOTES / REVISE / FAIL、重大度付きfindings、修正、残課題、
再開条件を記す。計画の品質と実装の完了を別判定にする。
