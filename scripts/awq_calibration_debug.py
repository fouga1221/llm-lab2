#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Iterable, List, Sequence

# デフォルト設定（ノートブックと同じ値を継承）
DEFAULT_REPO_DIR = Path(__file__).resolve().parents[1]

DEFAULT_CALIBRATION_SAMPLES = [
    " ".join(
        (
            "Calibration sample block A sentence {i}. "
            "This synthetic paragraph maintains diversity with instructions, status reports, "
            "and contextual hints about quantization workflows for large language models. "
            "It references AWQ calibration routines, tensor inspection, and fallback recovery paths "
            "while enumerating checkpoints such as step {i} and step {next_i}. "
            "By repeating rich vocabulary—metrics, regulators, assistants, deploy scripts—we emulate "
            "the varied prompts typically fed into tokenizer pipelines."
        ).format(i=i, next_i=i + 1)
        for i in range(1, 161)
    ),
    " ".join(
        (
            "Calibration sample block B sentence {i}. "
            "The passage describes data preprocessing, dataset audits, experiment tracking, "
            "and mixed precision validation suited for AutoAWQ instrumentation. "
            "It highlights retry logic, watchdog timers, safety valves, and telemetry exports "
            "to ensure the token stream exceeds the default 512 token threshold. "
            "Each iteration cites scenario labels {i}a and {next_i}b to broaden linguistic shape."
        ).format(i=i, next_i=i + 1)
        for i in range(1, 161)
    ),
]

DEFAULT_MAX_SEQ_LEN = 512  # AWQ 標準のシーケンス長に合わせる
DEFAULT_AWQ_GROUP_SIZE = 128


def configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def load_calibration_samples(
    file_path: Path | None,
    inline_samples: Iterable[str] | None,
) -> List[str]:
    samples: List[str] = list(DEFAULT_CALIBRATION_SAMPLES)

    if file_path:
        logging.info("外部ファイルからキャリブレーションサンプルを読み込みます: %s", file_path)
        text = file_path.read_text(encoding="utf-8")
        samples.extend(line.strip() for line in text.splitlines() if line.strip())

    if inline_samples:
        samples.extend(inline_samples)

    if not samples:
        raise ValueError("Calibration samples list is empty.")

    return samples


def import_prepare_awq(repo_root: Path):
    if not repo_root.exists():
        raise FileNotFoundError(f"Repository root does not exist: {repo_root}")

    repo_root_str = str(repo_root.resolve())
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
        logging.debug("sys.path にリポジトリルートを追加しました: %s", repo_root_str)

    try:
        module = importlib.import_module("src.llmlab.utils.awq")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "src.llmlab.utils.awq をインポートできませんでした。リポジトリルートの指定を確認してください。"
        ) from exc

    return module.prepare_awq_calib_data


def prepare_calibration_payload(
    prepare_fn,
    samples: Iterable[str],
    tokenizer,
):
    calib_data, calib_text_column = prepare_fn(samples)
    logging.info("キャリブレーションデータ種別: %s", type(calib_data).__name__)
    logging.info("テキスト列ヒント: %s", calib_text_column)

    total_tokens = 0
    if isinstance(calib_data, list) and calib_data:
        first = calib_data[0]
        if isinstance(first, str):
            for idx, sample in enumerate(calib_data):
                tokens = tokenizer.encode(sample, add_special_tokens=False)
                logging.info("Sample %d: %d token(s)", idx, len(tokens))
                total_tokens += len(tokens)
        elif isinstance(first, list):
            for idx, token_ids in enumerate(calib_data):
                logging.info("Sample %d: %d token(s)", idx, len(token_ids))
                total_tokens += len(token_ids)

    logging.info("キャリブレーション総トークン数: %d", total_tokens)
    return calib_data, calib_text_column


def build_llmcompressor_recipe(
    num_bits: int = 4,
    group_size: int = DEFAULT_AWQ_GROUP_SIZE,
) -> str:
    """
    llm-compressor のレシピ (AWQModifier) を YAML 文字列で生成する。
    """
    recipe = dedent(
        f"""
        modifiers:
          - AWQModifier:
              config_groups:
                group_0:
                  targets:
                    - Linear
                  weights:
                    num_bits: {num_bits}
                    type: int
                    symmetric: false
                    strategy: group
                    group_size: {group_size}
        """
    ).strip()
    return recipe


def write_calibration_jsonl(samples: Iterable[str], destination: Path) -> None:
    """
    llm-compressor の custom/json データセット向けに JSON Lines を生成する。
    """
    payload = [{"text": sample} for sample in samples]
    destination.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in payload) + "\n",
        encoding="utf-8",
    )


def run_llmcompressor_quantisation(
    model_name: str,
    tokenizer_name: str,
    samples: List[str],
    output_dir: Path | None,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    num_bits: int = 4,
    group_size: int = DEFAULT_AWQ_GROUP_SIZE,
) -> None:
    logging.info("llm-compressor で AWQ 量子化を実行します: %s", model_name)
    try:
        from llmcompressor.entrypoints.oneshot import oneshot as run_oneshot
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "llmcompressor がインポートできません。`pip install llmcompressor` を実行してください。"
        ) from exc

    recipe_yaml = build_llmcompressor_recipe(num_bits=num_bits, group_size=group_size)
    num_calibration = len(samples)
    logging.info("校正サンプル数 (llm-compressor 渡し): %d", num_calibration)

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = Path(tmpdir) / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_file = dataset_dir / "calibration.json"
        write_calibration_jsonl(samples, dataset_file)
        logging.debug("一時 JSON データセットを生成しました: %s", dataset_file)

        run_oneshot(
            model=model_name,
            tokenizer=tokenizer_name,
            trust_remote_code_model=True,
            precision="auto",
            recipe=recipe_yaml,
            dataset="json",
            dataset_path=str(dataset_dir),
            text_column="text",
            num_calibration_samples=num_calibration,
            max_seq_length=max_seq_len,
            output_dir=str(output_dir) if output_dir else None,
            save_compressed=True,
        )

    logging.info("llm-compressor による AWQ 量子化が完了しました。")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "AWQ キャリブレーションデータの変換と llm-compressor ベースの量子化デバッグを行うスクリプト。"
        )
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=DEFAULT_REPO_DIR,
        help="リポジトリルートのパス。通常は自動で設定される。",
    )
    parser.add_argument(
        "--tokenizer-name",
        default="Qwen/Qwen3-14B",
        help="デバッグ用に読み込むトークナイザ ID。",
    )
    parser.add_argument(
        "--awq-model-name",
        default=None,
        help="量子化対象のモデル ID。未指定時は --tokenizer-name と同じ値を使用。",
    )
    parser.add_argument(
        "--run-quant-test",
        action="store_true",
        default=False,
        help="llm-compressor を用いた AWQ 量子化テストを実行する。",
    )
    parser.add_argument(
        "--calibration-sample",
        action="append",
        help="キャリブレーション用テキストを追加指定する。複数指定可。",
    )
    parser.add_argument(
        "--calibration-samples-file",
        type=Path,
        help="1 行 1 サンプルでテキストを列挙したファイルパス。",
    )
    parser.add_argument(
        "--quant-output-dir",
        type=Path,
        default=None,
        help="量子化済みモデルを保存するディレクトリ。未指定時は保存をスキップ。",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="ログレベル（DEBUG/INFO/WARNING/...）。",
    )

    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    repo_root = args.repo_dir.resolve()
    prepare_fn = import_prepare_awq(repo_root)

    logging.info("リポジトリルート: %s", repo_root)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "transformers をインポートできません。必要なパッケージがインストールされているか確認してください。"
        ) from exc

    tokenizer_name = args.tokenizer_name
    awq_model_name = args.awq_model_name or tokenizer_name

    logging.info("トークナイザを読み込みます: %s", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    samples = load_calibration_samples(args.calibration_samples_file, args.calibration_sample)
    logging.info("キャリブレーションサンプル数: %d", len(samples))

    calib_data, calib_text_column = prepare_calibration_payload(prepare_fn, samples, tokenizer)

    if args.run_quant_test:
        output_dir = args.quant_output_dir
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        run_llmcompressor_quantisation(
            model_name=awq_model_name,
            tokenizer_name=tokenizer_name,
            samples=samples,
            output_dir=output_dir,
            max_seq_len=DEFAULT_MAX_SEQ_LEN,
            num_bits=4,
            group_size=DEFAULT_AWQ_GROUP_SIZE,
        )

    logging.info("処理が完了しました。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
