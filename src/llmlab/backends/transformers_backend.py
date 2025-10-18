# file: src/llmlab/backends/transformers_backend.py
from __future__ import annotations

import gc
import subprocess
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - オプショナル依存
    psutil = None  # type: ignore


class ModelBundle(TypedDict):
    """ロード済みモデルとトークナイザ、正規化済み設定を格納するバンドル。"""

    model: Any
    tok: Any
    cfg: Dict[str, Any]


_DEFAULT_GEN_KWARGS: Dict[str, Any] = {
    "max_new_tokens": 200,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.05,
}


def load_model(cfg: Dict[str, Any]) -> ModelBundle:
    """Transformers ベースの CausalLM を読み込み、量子化や LoRA を適用する。"""
    cfg_norm = dict(cfg)
    timings = dict(cfg_norm.get("_timings", {}))
    cfg_norm["_timings"] = timings

    model_name = cfg_norm.get("model_name") or cfg_norm.get("model_name_or_path")
    if not model_name:
        raise ValueError("model_name もしくは model_name_or_path を指定してください。")
    cfg_norm.setdefault("model_name", model_name)
    cfg_norm.setdefault("device_map", "auto")
    cfg_norm.setdefault("torch_dtype", "auto")
    cfg_norm.setdefault("quantization", "none")
    cfg_norm.setdefault("load_in_4bit", False)
    cfg_norm.setdefault("bnb_4bit_quant_type", "nf4")
    cfg_norm.setdefault("bnb_4bit_use_double_quant", True)
    cfg_norm.setdefault("max_seq_len", 4096)
    cfg_norm.setdefault("attn_implementation", None)
    cfg_norm.setdefault("trust_remote_code", True)

    quantization = str(cfg_norm.get("quantization", "none")).lower()
    torch_dtype = _resolve_dtype(cfg_norm.get("torch_dtype"))
    load_kwargs: Dict[str, Any] = {
        "device_map": cfg_norm["device_map"],
        "trust_remote_code": bool(cfg_norm.get("trust_remote_code", True)),
    }
    if cfg_norm.get("attn_implementation"):
        load_kwargs["attn_implementation"] = cfg_norm["attn_implementation"]
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype

    load_start = time.perf_counter()
    model, resolved_quant = _load_model_core(
        model_name=model_name,
        quantization=quantization,
        load_kwargs=load_kwargs,
        cfg=cfg_norm,
    )
    timings["load_model_s"] = time.perf_counter() - load_start
    cfg_norm["quantization"] = resolved_quant

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=load_kwargs["trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"
    model.eval()

    lora_path = cfg_norm.get("lora_path")
    merge_lora = bool(cfg_norm.get("merge_lora", False))
    cfg_norm["lora_applied"] = False
    timings["load_lora_s"] = 0.0
    if lora_path:
        lora_paths = (
            [lora_path]
            if isinstance(lora_path, str)
            else [path for path in lora_path if path]
        )
        if lora_paths:
            model, load_lora_s = _apply_lora_adapters(
                base_model=model,
                paths=lora_paths,
                merge=merge_lora,
            )
            if load_lora_s > 0.0:
                cfg_norm["lora_applied"] = True
            timings["load_lora_s"] = load_lora_s

    cfg_norm["dtype"] = str(model.dtype).replace("torch.", "")
    cfg_norm["max_seq_len_loaded"] = cfg_norm.get("max_seq_len")

    return ModelBundle(model=model, tok=tokenizer, cfg=cfg_norm)


def generate_texts(
    bundle: ModelBundle,
    prompts: List[str],
    **gen_kwargs: Any,
) -> List[str]:
    """指定したプロンプト群に対して新規生成テキストのみを返す。"""
    if not prompts:
        return []
    model = bundle["model"]
    tokenizer = bundle["tok"]
    if tokenizer is None:
        raise ValueError("Tokenizer がロードされていません。")

    kwargs = {**_DEFAULT_GEN_KWARGS, **gen_kwargs}
    stop_phrases = kwargs.pop("stop", None)
    eos_token_id = kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)
    kwargs.setdefault("pad_token_id", tokenizer.pad_token_id or eos_token_id)

    device = _infer_device(model)
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    )
    prompt_lengths = [len(ids) for ids in encoded["input_ids"]]
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        sequences = model.generate(**encoded, **kwargs)

    results: List[str] = []
    for seq, prompt_len in zip(sequences, prompt_lengths):
        gen_ids = seq[prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        text = _truncate_by_stop(text, stop_phrases)
        results.append(text)
    return results


def profile_generation(
    bundle: ModelBundle,
    prompts: List[str],
    **gen_kwargs: Any,
) -> Tuple[List[str], Dict[str, Any]]:
    """生成過程で時間・トークン数・メモリ指標などを計測する。"""
    tokenizer = bundle["tok"]
    model = bundle["model"]
    cfg = bundle["cfg"]

    if not prompts:
        return [], _build_metrics_stub(cfg, 0, 0, 0.0)
    if tokenizer is None:
        raise ValueError("Tokenizer がロードされていません。")

    tokens_in = _count_tokens(tokenizer, prompts)
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except AttributeError:  # pragma: no cover - 古いバージョン対策
            pass

    start = time.perf_counter()
    outputs = generate_texts(bundle, prompts, **gen_kwargs)
    gen_s = time.perf_counter() - start
    tokens_out = _count_tokens(tokenizer, outputs)

    gpu_mem_peak_mb = _get_gpu_peak_memory_mb()
    gpu_mem_used_mb = _get_gpu_runtime_memory_mb()
    cpu_rss_mb = _get_cpu_rss_mb()

    metrics = {
        "backend": "transformers",
        "model_name": cfg.get("model_name") or cfg.get("model_name_or_path"),
        "lora_path": cfg.get("lora_path"),
        "merged_lora": bool(cfg.get("merge_lora", False)),
        "quantization": cfg.get("quantization"),
        "dtype": cfg.get("dtype"),
        "max_seq_len": cfg.get("max_seq_len_loaded"),
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
    """ユーザー入力を受け取りながら停止語が入力されるまで応答を継続する。"""
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
    """内部参照を解除しガーベジコレクタと CUDA キャッシュを解放する。"""
    bundle["model"] = None
    bundle["tok"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_dtype(dtype_opt: Any) -> Optional[torch.dtype]:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_opt is None or str(dtype_opt).lower() == "auto":
        return None
    resolved = mapping.get(str(dtype_opt).lower())
    if resolved is None:
        warnings.warn(f"未知の torch_dtype {dtype_opt} が指定されたため auto を使用します。")
    return resolved


def _load_model_core(
    model_name: str,
    quantization: str,
    load_kwargs: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Tuple[Any, str]:
    quant_lower = quantization.lower()
    if quant_lower == "bnb4" or cfg.get("load_in_4bit"):
        model = _try_load_bnb4(model_name, load_kwargs, cfg)
        if model is not None:
            return model, "bnb4"
    if quant_lower == "gptq":
        model = _try_load_gptq(model_name, load_kwargs)
        if model is not None:
            return model, "gptq"
    if quant_lower == "awq":
        model = _try_load_awq(model_name, load_kwargs)
        if model is not None:
            return model, "awq"
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    return model, "none"


def _try_load_bnb4(
    model_name: str,
    load_kwargs: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Optional[Any]:
    try:
        from transformers import BitsAndBytesConfig  # type: ignore
    except ImportError:
        warnings.warn("bitsandbytes が見つからないため 4bit 量子化をスキップします。")
        return None
    kwargs = dict(load_kwargs)
    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
    )
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def _try_load_gptq(model_name: str, load_kwargs: Dict[str, Any]) -> Optional[Any]:
    try:
        from auto_gptq import AutoGPTQForCausalLM  # type: ignore
    except ImportError:
        warnings.warn("auto_gptq が見つからないため GPTQ 量子化をスキップします。")
        return None
    try:
        return AutoGPTQForCausalLM.from_quantized(model_name, **load_kwargs)
    except Exception as exc:  # pragma: no cover - 外部依存
        warnings.warn(f"GPTQ モデルの読み込みに失敗しました: {exc}")
        return None


def _try_load_awq(model_name: str, load_kwargs: Dict[str, Any]) -> Optional[Any]:
    try:
        from autoawq import AutoAWQForCausalLM  # type: ignore
    except ImportError:
        warnings.warn("autoawq が見つからないため AWQ 量子化をスキップします。")
        return None
    try:
        return AutoAWQForCausalLM.from_quantized(model_name, **load_kwargs)
    except Exception as exc:  # pragma: no cover - 外部依存
        warnings.warn(f"AWQ モデルの読み込みに失敗しました: {exc}")
        return None


def _apply_lora_adapters(
    base_model: Any,
    paths: List[str],
    merge: bool,
) -> Tuple[Any, float]:
    try:
        from peft import PeftModel  # type: ignore
    except ImportError:
        warnings.warn("peft が見つからないため LoRA を適用できません。")
        return base_model, 0.0

    current_model = base_model
    total_time = 0.0
    adapters_loaded = False

    for idx, path in enumerate(paths):
        start = time.perf_counter()
        try:
            if idx == 0:
                current_model = PeftModel.from_pretrained(current_model, path)
            else:
                load_adapter = getattr(current_model, "load_adapter", None)
                if callable(load_adapter):
                    load_adapter(path, adapter_name=f"lora_{idx}")
                else:
                    current_model = PeftModel.from_pretrained(current_model, path)
            adapters_loaded = True
        except Exception as exc:  # pragma: no cover - 外部依存
            warnings.warn(f"LoRA {path} の適用に失敗しました: {exc}")
            continue
        total_time += time.perf_counter() - start

    if not adapters_loaded:
        return base_model, 0.0

    if merge:
        merge_start = time.perf_counter()
        try:
            current_model = current_model.merge_and_unload()
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"LoRA のマージに失敗したため適用済みアダプタを保持します: {exc}")
        total_time += time.perf_counter() - merge_start

    if hasattr(current_model, "eval"):
        current_model.eval()
    return current_model, total_time


def _truncate_by_stop(text: str, stop_phrases: Optional[List[str]]) -> str:
    if not stop_phrases:
        return text
    lowered_text = text.lower()
    min_idx: Optional[int] = None
    for phrase in stop_phrases:
        idx = lowered_text.find(phrase.lower())
        if idx != -1 and (min_idx is None or idx < min_idx):
            min_idx = idx
    return text[:min_idx] if min_idx is not None else text


def _count_tokens(tokenizer: Any, texts: List[str]) -> Optional[int]:
    try:
        return sum(len(tokenizer.encode(t)) for t in texts)
    except Exception:  # pragma: no cover - tokenizer 仕様差異考慮
        return None


def _infer_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device) and device.type != "meta":
        return device
    try:
        return next(model.parameters()).device  # type: ignore[call-arg]
    except StopIteration:
        return torch.device("cpu")


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
    except Exception:  # pragma: no cover - CUDA 以外
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
        line = result.stdout.strip().splitlines()
        if not line:
            return None
        return float(line[0])
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
        "backend": "transformers",
        "model_name": cfg.get("model_name") or cfg.get("model_name_or_path"),
        "lora_path": cfg.get("lora_path"),
        "merged_lora": bool(cfg.get("merge_lora", False)),
        "quantization": cfg.get("quantization"),
        "dtype": cfg.get("dtype"),
        "max_seq_len": cfg.get("max_seq_len_loaded"),
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
