# file: src/llmlab/backends/vllm_backend.py
from __future__ import annotations

import gc
import subprocess
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import torch
from vllm import LLM, SamplingParams

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - オプショナル依存
    psutil = None  # type: ignore


class ModelBundle(TypedDict):
    """vLLM ランタイムの LLM 本体とトークナイザ、正規化済み設定のバンドル。"""

    llm: Any
    tok: Any
    cfg: Dict[str, Any]


def load_model(cfg: Dict[str, Any]) -> ModelBundle:
    """vLLM エンジンを初期化し、必要に応じて LoRA を適用する。"""
    cfg_norm = dict(cfg)
    timings = dict(cfg_norm.get("_timings", {}))
    cfg_norm["_timings"] = timings

    model_name = cfg_norm.get("model_name") or cfg_norm.get("model_name_or_path")
    if not model_name:
        raise ValueError("model_name もしくは model_name_or_path を指定してください。")
    cfg_norm.setdefault("model_name", model_name)
    cfg_norm.setdefault("tensor_parallel_size", 1)
    cfg_norm.setdefault("dtype", "auto")
    cfg_norm.setdefault("quantization", "none")
    cfg_norm.setdefault("lora_applied", False)

    llm_kwargs: Dict[str, Any] = {"model": model_name}
    if cfg_norm.get("tensor_parallel_size"):
        llm_kwargs["tensor_parallel_size"] = int(cfg_norm["tensor_parallel_size"])
    dtype = cfg_norm.get("dtype")
    if dtype and str(dtype).lower() != "auto":
        llm_kwargs["dtype"] = dtype
    max_len = cfg_norm.get("max_model_len") or cfg_norm.get("max_seq_len")
    if max_len:
        llm_kwargs["max_model_len"] = int(max_len)
    if cfg_norm.get("gpu_memory_utilization") is not None:
        llm_kwargs["gpu_memory_utilization"] = float(cfg_norm["gpu_memory_utilization"])
    if cfg_norm.get("download_dir"):
        llm_kwargs["download_dir"] = cfg_norm["download_dir"]
    if cfg_norm.get("quantization") and str(cfg_norm["quantization"]).lower() != "none":
        llm_kwargs["quantization"] = cfg_norm["quantization"]
    if cfg_norm.get("trust_remote_code") is not None:
        llm_kwargs["trust_remote_code"] = bool(cfg_norm["trust_remote_code"])
    if cfg_norm.get("enforce_eager") is not None:
        llm_kwargs["enforce_eager"] = bool(cfg_norm["enforce_eager"])

    load_start = time.perf_counter()
    llm = LLM(**llm_kwargs)
    timings["load_model_s"] = time.perf_counter() - load_start

    tokenizer = None
    try:
        tokenizer = llm.get_tokenizer()
    except Exception:  # pragma: no cover - バージョン差吸収
        tokenizer = None

    lora_path = cfg_norm.get("lora_path")
    merge_lora = bool(cfg_norm.get("merge_lora", False))
    timings["load_lora_s"] = 0.0
    if lora_path:
        lora_paths = (
            [lora_path]
            if isinstance(lora_path, str)
            else [path for path in lora_path if path]
        )
        if lora_paths:
            load_lora_s, applied = _apply_vllm_lora(llm, lora_paths, merge_lora)
            timings["load_lora_s"] = load_lora_s
            cfg_norm["lora_applied"] = applied

    cfg_norm["tok_available"] = tokenizer is not None
    cfg_norm["quantization"] = cfg_norm.get("quantization", "none")

    return ModelBundle(llm=llm, tok=tokenizer, cfg=cfg_norm)


def generate_texts(
    bundle: ModelBundle,
    prompts: List[str],
    **gen_kwargs: Any,
) -> List[str]:
    """vLLM のバッチ生成を行い、新規生成テキストのみを返す。"""
    if not prompts:
        return []
    llm = bundle["llm"]
    sampling_params = _build_sampling_params(gen_kwargs)
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    results: List[str] = []
    for request_output in outputs:
        if not request_output.outputs:
            results.append("")
            continue
        results.append(request_output.outputs[0].text)
    return results


def profile_generation(
    bundle: ModelBundle,
    prompts: List[str],
    **gen_kwargs: Any,
) -> Tuple[List[str], Dict[str, Any]]:
    """生成結果と併せてメトリクス（時間・トークン数・メモリなど）を返す。"""
    tokenizer = bundle["tok"]
    cfg = bundle["cfg"]
    llm = bundle["llm"]

    if not prompts:
        return [], _build_metrics_stub(cfg, 0, 0, 0.0)

    tokens_in = _count_tokens(tokenizer, prompts)
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except AttributeError:  # pragma: no cover
            pass

    start = time.perf_counter()
    outputs = generate_texts(bundle, prompts, **gen_kwargs)
    gen_s = time.perf_counter() - start
    tokens_out = _count_tokens(tokenizer, outputs)

    gpu_mem_peak_mb = _get_gpu_peak_memory_mb()
    gpu_mem_used_mb = _get_gpu_runtime_memory_mb()
    cpu_rss_mb = _get_cpu_rss_mb()

    metrics = {
        "backend": "vllm",
        "model_name": cfg.get("model_name") or cfg.get("model_name_or_path"),
        "lora_path": cfg.get("lora_path"),
        "merged_lora": bool(cfg.get("merge_lora", False)),
        "quantization": cfg.get("quantization"),
        "dtype": cfg.get("dtype"),
        "max_seq_len": cfg.get("max_model_len") or cfg.get("max_seq_len"),
        "n_prompts": len(prompts),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "load_model_s": cfg["_timings"].get("load_model_s"),
        "load_lora_s": cfg["_timings"].get("load_lora_s", 0.0),
        "gen_s": gen_s,
        "tok_s_out": (tokens_out / gen_s) if gen_s > 0 and tokens_out is not None else None,
        "gpu_mem_peak_mb": gpu_mem_peak_mb,
        "gpu_mem_used_mb": gpu_mem_used_mb,
        "cpu_rss_mb": cpu_rss_mb,
    }
    return outputs, metrics


def chat_loop(
    bundle: ModelBundle,
    system_prompt: Optional[str] = None,
    stop_phrases: Optional[List[str]] = None,
    **gen_kwargs: Any,
) -> None:
    """停止語が入力されるまで対話を継続する簡易ループ。"""
    history: List[Dict[str, str]] = []
    stop_set = {phrase.lower() for phrase in (stop_phrases or ["/exit", ":q", "終了"])}
    sys_prompt = system_prompt or ""

    while True:
        user_input = input("You> ").strip()
        if not user_input:
            continue
        if user_input.lower() in stop_set:
            break

        history.append({"role": "user", "content": user_input})
        prompt = _build_chat_prompt(sys_prompt, history)
        reply = generate_texts(bundle, [prompt], **gen_kwargs)[0]
        history.append({"role": "assistant", "content": reply})
        print(f"Assistant> {reply}")


def free_model(bundle: ModelBundle) -> None:
    """LLM 参照を外し、メモリを回収する。"""
    bundle["llm"] = None
    bundle["tok"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_sampling_params(gen_kwargs: Dict[str, Any]) -> SamplingParams:
    params = dict(gen_kwargs)
    stop = params.pop("stop", None)
    max_new = params.pop("max_new_tokens", None)
    if max_new is not None and "max_tokens" not in params:
        params["max_tokens"] = int(max_new)
    cleaned = {k: v for k, v in params.items() if v is not None}
    sampling_kwargs = dict(cleaned)
    if stop is not None:
        sampling_kwargs["stop"] = stop
    try:
        return SamplingParams(**sampling_kwargs)
    except TypeError as exc:
        invalid_keys = ", ".join(sorted(sampling_kwargs.keys()))
        raise ValueError(f"SamplingParams 生成に失敗しました: {exc} (keys: {invalid_keys})") from exc


def _apply_vllm_lora(
    llm: Any,
    paths: List[str],
    merge: bool,
) -> Tuple[float, bool]:
    adapter_names: List[str] = []
    total_time = 0.0
    applied = False

    for idx, path in enumerate(paths):
        adapter_name = f"lora_{idx}"
        start = time.perf_counter()
        try:
            if hasattr(llm, "load_lora_adapter"):
                load_fn = llm.load_lora_adapter
                try:
                    load_fn(path, adapter_name=adapter_name)
                except TypeError:
                    load_fn(adapter_name, path)
                applied = True
            elif hasattr(llm, "add_adapter"):
                llm.add_adapter(path, adapter_name=adapter_name)
                applied = True
            elif hasattr(llm, "apply_lora"):
                llm.apply_lora(path)
                adapter_names = []
                applied = True
                break
            else:
                warnings.warn("vLLM の LoRA API が見つからないため適用をスキップします。")
                break
        except Exception as exc:  # pragma: no cover - 外部 API 差異
            warnings.warn(f"LoRA {path} の適用に失敗しました: {exc}")
            continue
        total_time += time.perf_counter() - start
        adapter_names.append(adapter_name)

    if applied and adapter_names and hasattr(llm, "set_active_adapters"):
        try:
            llm.set_active_adapters(adapter_names)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"LoRA アダプタの有効化に失敗しました: {exc}")

    if applied and merge:
        start = time.perf_counter()
        merged = False
        for attr in ("merge_lora_adapters", "merge_adapter"):
            merge_fn = getattr(llm, attr, None)
            if callable(merge_fn):
                try:
                    merge_fn()
                    merged = True
                    break
                except Exception as exc:  # pragma: no cover
                    warnings.warn(f"LoRA マージに失敗しました: {exc}")
        if not merged:
            warnings.warn("LoRA マージ API が利用できないためスキップします。")
        total_time += time.perf_counter() - start

    return total_time, applied


def _count_tokens(tokenizer: Any, texts: List[str]) -> Optional[int]:
    if tokenizer is None:
        return None
    try:
        return sum(len(tokenizer.encode(t)) for t in texts)
    except Exception:  # pragma: no cover - tokenizer 差異考慮
        return None


def _build_chat_prompt(system_prompt: str, history: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    if system_prompt:
        lines.append(f"System: {system_prompt}")
    for turn in history:
        prefix = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{prefix}: {turn['content']}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _get_gpu_peak_memory_mb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:  # pragma: no cover
        return None


def _get_gpu_runtime_memory_mb() -> Optional[float]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = result.stdout.strip().splitlines()
        if not lines:
            return None
        return float(lines[0])
    except Exception:  # pragma: no cover - GPU 非搭載環境
        return None


def _get_cpu_rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except Exception:  # pragma: no cover - 環境依存
        return None


def _build_metrics_stub(
    cfg: Dict[str, Any],
    tokens_in: Optional[int],
    tokens_out: Optional[int],
    gen_s: float,
) -> Dict[str, Any]:
    return {
        "backend": "vllm",
        "model_name": cfg.get("model_name") or cfg.get("model_name_or_path"),
        "lora_path": cfg.get("lora_path"),
        "merged_lora": bool(cfg.get("merge_lora", False)),
        "quantization": cfg.get("quantization"),
        "dtype": cfg.get("dtype"),
        "max_seq_len": cfg.get("max_model_len") or cfg.get("max_seq_len"),
        "n_prompts": 0,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "load_model_s": cfg["_timings"].get("load_model_s"),
        "load_lora_s": cfg["_timings"].get("load_lora_s", 0.0),
        "gen_s": gen_s,
        "tok_s_out": None,
        "gpu_mem_peak_mb": _get_gpu_peak_memory_mb(),
        "gpu_mem_used_mb": _get_gpu_runtime_memory_mb(),
        "cpu_rss_mb": _get_cpu_rss_mb(),
    }
