# AWQ キャリブレーションデバッグスクリプト

`scripts/awq_calibration_debug.py` は、従来ノートブックで行っていた AWQ キャリブレーション前処理をローカル環境で手軽に検証するためのツールです。校正サンプルの整形に加えて、llm-compressor を用いた AWQ 量子化ワークフローの動作確認も行えます。

## 使い方

```bash
python scripts/awq_calibration_debug.py \
  --tokenizer-name facebook/opt-125m \
  --run-quant-test
```

事前に `pip install "transformers>=4.56.0,<4.57.0" "llmcompressor>=0.8.1" "accelerate>=1.6.0,<1.11.0" safetensors datasets` など、ノートブックと同等の依存パッケージをインストールしておいてください。サンプルのトークン化結果はログへ出力され、最小トークン数要件を満たしているか確認できます。

- キャリブレーションサンプルを追加する場合は `--calibration-sample "テキスト"` を複数回指定するか、`--calibration-samples-file samples.txt` で 1 行 1 サンプルのファイルを渡せます。
- `--run-quant-test` を指定すると llm-compressor の `AWQModifier` を用いて量子化テストを実行し、`src/llmlab/utils/awq.prepare_awq_calib_data` による前処理結果が量子化に渡せるか確認できます。
- 量子化成果物を保存したい場合は `--quant-output-dir output/dir` を指定してください（未指定時は保存をスキップします）。

> **補足**: デフォルトのキャリブレーションサンプルは 512 トークンを大きく超える長文を用意しているため、そのまま `--run-quant-test` を実行しても llm-compressor の AWQ キャリブレーション要件を満たします。

## 主なオプション

| オプション | 説明 | デフォルト |
|------------|------|-------------|
| `--tokenizer-name` | 想定するトークナイザ（HF ID） | `facebook/opt-125m` |
| `--awq-model-name` | 量子化対象のモデル ID。未指定ならトークナイザと同じ値。 | - |
| `--run-quant-test` | llm-compressor による量子化を実行する | 無効 |
| `--calibration-sample` | キャリブレーションテキストを追加指定（複数回可） | - |
| `--calibration-samples-file` | キャリブレーションテキストを列挙したファイルパス | - |
| `--quant-output-dir` | 量子化済みモデルの保存先ディレクトリ | 保存なし |

## ログ

`--log-level DEBUG` を指定すると、トークナイザでのトークン数や内部処理の詳細がより多く出力され、ノートブックと同等のデバッグ情報を得られます。
