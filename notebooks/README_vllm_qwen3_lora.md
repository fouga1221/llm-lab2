# vLLM Qwen3-14B ノートブックガイド

本リポジトリの Colab 向けノートブックは、役割ごとに以下の 3 つへ分割しました。  
それぞれのノートブックは Google Drive との連携・依存パッケージの導入・設定値の永続化ロジックを共通化しており、ランタイム再起動後も設定セルを再実行するだけで作業を再開できます。

| ノートブック | 主な機能 | 備考 |
|--------------|----------|------|
| `finetuning_lora_qlora.ipynb` | LoRA / QLoRA によるファインチューニング | ベースモデル・データセット・学習ハイパーパラメータを設定し、LoRA アダプタを保存。任意でマージも可能。 |
| `quantization_awq_lora.ipynb` | LoRA マージ + llmcompressor AWQ 量子化 | LoRA マージセルと AWQ 量子化セルを分離。LoRA なしでの量子化もこのノートブックで実施。 |
| `inference_chat_vllm.ipynb` | vLLM 推論・ベンチマーク・チャットループ | 量子化結果またはベースモデルを読み込み、ベンチマークとチャットログ収集を実行。LoRA 切り替え機能付き。 |

---

## 共通の準備フロー

1. **セル1: Colab ユーティリティ**  
   - リポジトリの clone と `pip` ラッパ関数を定義し、必要なパッケージをインストールします。  
   - ランタイム再起動が必要になった場合はセル1とセル2を再実行してください。
2. **セル2: 設定と永続化**  
   - Google Drive をマウントし、`/content/llm-lab-save` へのシンボリックリンクを作成。  
   - 各種パス・ハイパーパラメータを JSON で保存し、再起動後も自動復元します。  
   - 設定を変更したら `persist_config(CONFIG)` を呼び出して保存します。

---

## ノートブック別サマリ

### 1. `finetuning_lora_qlora.ipynb`
- **セル3**: データセットの読み込みと整形。`DATASET_FORMAT` で JSONL / Hugging Face Dataset に対応。  
- **セル4**: LoRA / QLoRA ファインチューニング。`USE_QLORA` の切り替えや `TRAINING_ARGS` の調整が可能。  
- **セル5**: 任意で LoRA アダプタをベースモデルへマージ（CPU デフォルト）。  
- **セル6**: GPU メモリを開放する後片付け。

### 2. `quantization_awq_lora.ipynb`
- **セル3**: LoRA マージ処理。`ENABLE_LORA_MERGE=False` でスキップし、既存マージ済みモデルを再利用できます。  
- **セル4**: llmcompressor を用いた AWQ 量子化。キャリブレーションサンプルはリスト、ファイル、または Dataset ID で指定可能。  
- 量子化後は `AWQ_OUTPUT_DIR` に safetensors と `awq_config.json` を自動配置します。

### 3. `inference_chat_vllm.ipynb`
- **セル3**: vLLM モデルをロード。`reload_bundle()` で設定値に応じて再読み込みし、LoRA 適用の有無を永続化します。  
- **セル4**: `profile_generation` によるベンチマーク。生成結果とメトリクスをファイルへ保存。  
- **セル5**: `switch_lora()` ヘルパーで LoRA / 量子化設定を切り替え可能。  
- **セル6**: チャットループ。会話ログは JSONL 形式で `CHAT_LOG_PATH` に追記されます。  
- **セル7**: vLLM リソースの解放。

---

## 推奨ワークフロー例

1. **高性能 GPU (A100 等)** で `finetuning_lora_qlora.ipynb` を実行し、LoRA アダプタを作成。  
2. 同環境、もしくは十分な VRAM を持つ環境で `quantization_awq_lora.ipynb` を実行し、マージ済みモデルを AWQ 量子化。  
3. 推論専用環境（L4 / T4 等）に切り替えて `inference_chat_vllm.ipynb` を実行。  
   - `MODEL_PATH` に AWQ 出力フォルダを指定してベンチマーク・チャットを実施。  
   - LoRA を切り替えたい場合は `switch_lora()` を使用。

---

## 注意事項

- ノートブックのセルは依存順になっています。指定と異なる順序で実行しないでください。  
- 量子化やベンチマーク処理は GPU メモリを多く消費します。必要に応じて `tensor_parallel_size` や `gpu_memory_utilization` を調整してください。  
- 旧来の一体型ノートブック (`vllm_qwen3_lora_autoAWQ.ipynb` 等) はメンテナンス対象外です。新しい分割ノートブックへの移行を推奨します。

