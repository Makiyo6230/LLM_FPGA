import json
import os
import threading
import time
from pathlib import Path

import torch


_LOCK = threading.Lock()
_FORWARD_STEP = -1


def _is_enabled() -> bool:
    return os.environ.get("QWEN2_FPGA_RECORD", "").lower() in {"1", "true", "yes", "on"}


def _record_root() -> Path:
    return Path(os.environ.get("QWEN2_FPGA_RECORD_DIR", "./qwen2_fpga_records")).expanduser()


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _step_dir() -> Path:
    return _record_root() / f"step_{max(_FORWARD_STEP, 0):04d}"


def start_forward(name: str) -> int | None:
    if not _is_enabled():
        return None
    try:
        global _FORWARD_STEP
        with _LOCK:
            _FORWARD_STEP += 1
            step = _FORWARD_STEP
        record_metadata(
            "forward",
            {
                "step": step,
                "name": name,
                "time": time.time(),
            },
        )
        return step
    except Exception:
        return None


def record_metadata(name: str, payload: dict) -> None:
    if not _is_enabled():
        return
    try:
        root = _step_dir()
        root.mkdir(parents=True, exist_ok=True)
        metadata_path = root / "metadata.jsonl"
        item = {"name": name, **payload}
        with _LOCK:
            with metadata_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    except Exception:
        return


def record_tensor(name: str, tensor: torch.Tensor | None, layer_idx: int | None = None) -> None:
    if not _is_enabled() or tensor is None or not isinstance(tensor, torch.Tensor):
        return
    try:
        root = _step_dir()
        if layer_idx is not None:
            root = root / f"layer_{layer_idx:02d}"
        root.mkdir(parents=True, exist_ok=True)

        cpu_tensor = tensor.detach().contiguous().cpu()
        original_dtype = str(cpu_tensor.dtype)
        stored_dtype = original_dtype
        if cpu_tensor.dtype == torch.bfloat16:
            cpu_tensor = cpu_tensor.to(torch.float32)
            stored_dtype = "torch.float32_from_bfloat16"

        file_name = f"{_safe_name(name)}.npy"
        file_path = root / file_name

        import numpy as np

        np.save(file_path, cpu_tensor.numpy())
        record_metadata(
            "tensor",
            {
                "tensor_name": name,
                "layer_idx": layer_idx,
                "file": str(file_path),
                "shape": list(tensor.shape),
                "dtype": original_dtype,
                "stored_dtype": stored_dtype,
                "device": str(tensor.device),
            },
        )
    except Exception:
        return
