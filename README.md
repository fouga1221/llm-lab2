# llm-lab2

シンプルな Qwen3-14B 向け LoRA/QLoRA → マージ → AWQ 量子化 → vLLM サerving → 単発ベンチパイプライン。

## Colab Notebook エントリーポイント
下記バッジをクリックすると本リポジトリの統合 Notebook (`notebooks/pipeline.ipynb`) を Colab 上で直接開けます。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fouga1221/llm-lab2/blob/main/notebooks/pipeline.ipynb)

Notebook 概要:
- Parameter Block (大文字変数) による設定一元化 & `config_hash` 自動生成
- データロード → 前処理 → LoRA/QLoRA 学習 → 推論テスト → ベンチ (TTFT/Throughput/VRAM) → Perplexity → マージ → 連続対話/ベンチ付き対話
- `chat_loop_bench()` により対話各ターンの TTFT / tok_per_s を CSV 追記

利用手順 (Colab 上):
1. バッジで開く → 「ドライブにコピー」を選択 (元リポは読み取り専用)
2. セル "4a. パス手動設定" で `DATA_ROOT` / `CSV_PATH` / `JSONL_PATH` を自身の環境に合わせて記入
3. 上から順に実行 (学習不要でベンチのみ試す場合は 12 以降を調整)
4. 実行結果は `results/run_YYYYMMDD_HHMMSS/` 配下 (loss_curve, bench.csv, metrics.json, report.json 等)

ローカル GPU で使う場合: Notebook をダウンロードし `pip install -r requirements.txt` 後 `jupyter lab` / VS Code で開いて同様に実行してください。

### 推論専用 Notebook エントリーポイント
学習済み (もしくは HF Hub 公開) モデル + LoRA アダプタを用いて高速にベンチ付き対話推論を行う `notebooks/inference.ipynb` も利用できます。

[![Open In Colab - Inference](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fouga1221/llm-lab2/blob/main/notebooks/inference.ipynb)

特徴:
- ローカルパス / HF モデルID 自動判定 (ローカル無ければ Hub 参照)
- LoRA / マージ済みモデルの両モード対応 (`LOAD_MODE = 'base+lora' | 'merged'`)
- L4 GPU 向け 4bit 省メモリ推奨プリセット (TTFT ストリーミング計測)
- 連続対話ループ: 各ターンの `ttft_ms, gen_ms, total_ms, tokens_per_s, prompt_tokens, new_tokens, mem_*` を `bench.csv` に追記
- セッション終了後 `session_summary.json` / `conversation.json` / `params_{config_hash}.json` を保存

利用手順 (Colab):
1. バッジで開き「ドライブにコピー」
2. 先頭の GPU 検出セルを実行し推奨設定を確認 (L4 以外の場合は PRECISION_MODE を適宜調整)
3. パラメータブロックで `LOAD_MODE` / モデル参照 (`*_PATH` or `*_REF`) / 生成長 (`GEN_KW.max_new_tokens`) を編集
4. YAML プロンプトを使う場合は `PROMPT_SOURCE='yaml'` とし `PROMPT_YAML_PATH` を配置
5. 連続対話セル (#9) を実行して `/exit` で終了
6. 終了後 #10 サマリセルを実行しメトリクス確認

小ネタ / TIPS:
- 4bit で品質懸念がある場合: `PRECISION_MODE='auto'` + マージ済み bf16 モデルを参照
- 長対話で VRAM が増える場合: `TRUNCATE_PROMPT_TOKENS` を 3072 などに下げる
- TTFT 改善をさらに可視化したい場合: 対話ループを短いプロンプト (例: "ping") で複数回繰り返し p95 を確認

---

## セットアップ (推奨パッケージ)
```
pip install torch transformers peft trl bitsandbytes llmcompressor autoawq vllm
```

## 1. QLoRA (または LoRA) 学習
```
python scripts/train_qlora.py --base_model Qwen/Qwen2.5-14B \
	--data_jsonl data/train.jsonl \
	--output_dir runs/qlora-adapter
```
LoRA (4bit無効) の場合は `--no_qlora` を付与。

## 2. LoRA マージ
```
python scripts/merge_lora.py --base_model Qwen/Qwen2.5-14B \
	--adapter_dir runs/qlora-adapter \
	--out_model_dir runs/merged-fp16
```

## 3. AWQ 量子化 (llmcompressor 優先)
```
python scripts/quant_awq.py --model_dir runs/merged-fp16 --out_dir runs/awq-w4a16 --fallback_autoawq
```

## 4. vLLM サーバ起動
```
python scripts/serve_vllm.py --model runs/awq-w4a16 --port 8000 --max_model_len 4096
```

### 4.1 量子化オプション指定
```
python scripts/serve_vllm.py --model runs/awq-w4a16 --port 8000 \
	--max_model_len 4096 --quantization awq
```
`--quantization` を付与すると vLLM に量子化種類 (awq / gptq / fp8 など対応範囲) を明示できます。

### 4.2 オフライン起動
事前にモデル/量子化重みをローカルへ格納済の場合:
```
python scripts/serve_vllm.py --model runs/awq-w4a16 --port 8000 \
	--max_model_len 4096 --quantization awq --offline --download_dir ./.hf_cache
```
内部で `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` を設定。`--download_dir` を指定すると HF_HOME をそのディレクトリに誘導し、完全オフラインで起動可能 (依存する tokenizer/config も同ディレクトリ配下に存在する前提)。

## 5. 単発ベンチマーク
Transformers (fp16 merged):
```
python scripts/bench_single.py --mode transformers --model runs/merged-fp16 \
	--prompt_file samples/prompt.txt --csv results/bench.csv --quant none
```
vLLM (AWQ):
```
python scripts/bench_single.py --mode vllm \
	--endpoint http://127.0.0.1:8000/v1/chat/completions \
	--model_name runs/awq-w4a16 \
	--prompt_file samples/prompt.txt \
	--csv results/bench.csv \
	--quant awq
```

## 6. CSV カラム
`runtime,quant,prompt_len,prompt_tokens,max_new,ttft_ms,e2e_ms,decode_ms,out_tokens,total_tokens,ms_per_tok,tok_per_s,vram_peak_gb,commit_sha,versions,config_hash,seed`

## 7. 再現性シード
`GLOBAL_SEED` 環境変数 or `--seed` 引数で指定。内部で python, numpy, torch CUDA を統一初期化。

## TODO / 改善余地
- 実データに基づく AWQ キャリブレーション
- ベンチ複数ケース一括実行スクリプト
- vLLM out_tokens: chunk毎 tokenizer での真値 token 増分計測 (現状最終テキスト一括)
- 追加メトリクス: GPU名 / driver / peak alloc vs reserved 比率

## 免責
研究用サンプル。実運用では監視・ログ強化, エラーハンドリング追加を推奨。