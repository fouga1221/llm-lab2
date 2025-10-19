# vLLM Qwen3-14B LoRA / AWQ Notebook 使い方メモ

`notebooks/vllm_qwen3_lora.ipynb` は、LoRA 適用済み Qwen3-14B を

1. CPU 上で LoRA をマージ  
2. AWQ で量子化  
3. vLLM で推論

まで一気通貫で行うことを想定したノートブックです。  
以下では Colab 環境（A100 → L4 切り替え）の利用例を中心に、
主要セルの設定とワークフローをまとめます。

---

## 事前準備

* Google Drive を `/content/drive` にマウント（標準機能）。  
* `DATA_ROOT` 直下に成果物が書き込まれるため、十分な空き容量を確保する。  
* base モデル / LoRA アダプタ / AWQ 成果物は Drive 上で共有しておくと復旧が容易。  
* ノートブックは Colab から以下のリンクで開けます：  
  [Open in Colab](https://colab.research.google.com/github/fouga1221/llm-lab2/blob/main/notebooks/vllm_qwen3_lora.ipynb)

---

## セル構成と主要設定

### セル1: 環境構築
- `REPO_DIR` … clone 先を変更したい場合のみ編集。  
- `REPO_BRANCH` … 特定ブランチを使うなら変更。  
- `BASE_PACKAGES` … 必要パッケージ。Colab GPU であればそのままで OK。

### セル2: 定数設定
- `CONFIG["BASE_MODEL_NAME"]` … LoRA のベースモデル ID。  
- `CONFIG["LORA_DIR"]` … LoRA アダプタの配置先。  
- `CONFIG["MERGED_OUTPUT_DIR"]` / `CONFIG["AWQ_OUTPUT_DIR"]` … デフォルトの出力先。  
- `CONFIG["EXTERNAL_MERGED_DIR"]` / `CONFIG["EXTERNAL_AWQ_DIR"]` … すでに成果物がある場合に指定するパス。  
- `CONFIG["PERFORM_LORA_MERGE"]` / `CONFIG["PERFORM_AWQ_QUANT"]`  
  - `True` … セル3でマージ／量子化を実行する。  
  - `False` … セル3は既存成果物（`EXTERNAL_*` または既定ディレクトリ）を再利用する。

### セル3: LoRA マージ & AWQ 量子化
- A100 など潤沢な GPU で実行 → 成果物を Drive に保存 → 切り替え先で再利用。  
- `PERFORM_*` が `False` の場合は `EXTERNAL_*` または既定ディレクトリから成果物を検出し、`VLLM_MODEL_PATH` と `VLLM_QUANTIZATION` を自動設定。

### セル5 以降
- セル5: vLLM ロード（`cfg["model_name"]` はセル3で決定済み）  
- セル6: バッチ推論とメトリクス集計  
- セル7: 対話ループ（`/exit` で終了）  
- セル8: 後片付け（`free_model`）

---

## 推奨ワークフロー

### 1. A100 など大きめ GPU で準備
1. セル1～2を実行。  
2. `CONFIG["PERFORM_LORA_MERGE"] = True`、`CONFIG["PERFORM_AWQ_QUANT"] = True` のままセル3を実行。  
3. 生成された `MERGED_OUTPUT_DIR` / `AWQ_OUTPUT_DIR` を Drive 等に保存。  
4. セル4以降は任意（動作確認したければ実行）。

### 2. L4 / T4 等で推論のみ行う
1. 同じノートブックを開きセル1～2を実行。  
2. `CONFIG["PERFORM_LORA_MERGE"] = False`、`CONFIG["PERFORM_AWQ_QUANT"] = False` に設定。  
3. `CONFIG["EXTERNAL_AWQ_DIR"]`（必要なら `EXTERNAL_MERGED_DIR`）に A100 で作成した成果物パスを指定。  
4. セル3を実行すると既存成果物が検出される。以降セル5～8で推論と対話を実行。

---

## メモ
- 14B モデルの LoRA マージは一時的に 30GB 近い RAM を消費するため、A100 クラスの環境推奨。Colab 標準（L4/T4）では途中でプロセスが落ちるケースが多い。  
- AWQ 量子化も CPU で実行可能だが時間を要する。GPU があれば `device_map={\"\": \"cuda\"}` に変更すると高速化可。  
- 生成された成果物は Drive にバックアップしておくと環境切り替えが容易。  
- 既存成果物を使う際は `EXTERNAL_*` でパスを指定し、`PERFORM_*` を `False` にするだけでよい。
