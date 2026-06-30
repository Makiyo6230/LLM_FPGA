"""Run Qwen2.5-0.5B-Instruct generation and export FPGA golden trace.

This script is the clean entry point for reproducing the existing golden trace.
It expects the instrumented Transformers source to be installed or available on
PYTHONPATH. The instrumentation is controlled by:

  QWEN2_FPGA_RECORD=1
  QWEN2_FPGA_RECORD_DIR=<output directory>

Example on H200:

  export PYTHONNOUSERSITE=1
  unset PYTHONUSERBASE
  export QWEN2_FPGA_RECORD=1
  /data3/output/conda_envs/qwen25_fpga_transformers/bin/python \
    run_full_generate_trace.py \
    --model-path /data3/models/Qwen2.5-0.5B-Instruct \
    --record-dir /data3/output/qwen25_fpga_golden/cases/generate_reproduce \
    --max-new-tokens 8
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPT = "\u4f60\u597d\uff0c\u8bf7\u7528\u4e00\u53e5\u8bdd\u4ecb\u7ecd\u4f60\u81ea\u5df1\u3002"
DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Qwen2.5-0.5B full generation trace.")
    parser.add_argument("--model-path", default="/data3/models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--record-dir", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def save_generation_scores(record_dir: Path, scores: tuple[torch.Tensor, ...]) -> list[list[int]]:
    score_dir = record_dir / "generation_token_vocab_logits"
    score_dir.mkdir(parents=True, exist_ok=True)

    shapes: list[list[int]] = []
    for index, score in enumerate(scores):
        array = score.detach().cpu().to(torch.float32).numpy()
        if index == 0:
            name = "prefill_next_token_vocab_logits.npy"
        else:
            name = f"decode_step_{index:04d}_next_token_vocab_logits.npy"
        np.save(score_dir / name, array)
        shapes.append(list(array.shape))
    return shapes


def main() -> None:
    args = parse_args()
    record_dir = Path(args.record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)

    os.environ["QWEN2_FPGA_RECORD"] = "1"
    os.environ["QWEN2_FPGA_RECORD_DIR"] = str(record_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([chat_text], return_tensors="pt")
    input_ids = model_inputs.input_ids[0].tolist()
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation="eager",
    ).to(args.device)
    model.eval()

    model_inputs = model_inputs.to(args.device)
    input_length = int(model_inputs.input_ids.shape[1])

    with torch.no_grad():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

    sequence = generated.sequences[0]
    generated_ids = sequence[input_length:].detach().cpu().tolist()
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(sequence.detach().cpu().tolist(), skip_special_tokens=False)
    score_shapes = save_generation_scores(record_dir, tuple(generated.scores))

    step_dirs = sorted(path.name for path in record_dir.iterdir() if path.is_dir() and path.name.startswith("step_"))
    file_count = sum(1 for path in record_dir.rglob("*") if path.is_file())

    metadata = {
        "model_path": args.model_path,
        "record_dir": str(record_dir),
        "prompt": args.prompt,
        "messages": messages,
        "chat_text": chat_text,
        "input_token_count": len(input_ids),
        "input_ids": input_ids,
        "input_tokens": input_tokens,
        "generated_token_count": len(generated_ids),
        "generated_ids": generated_ids,
        "generated_tokens": generated_tokens,
        "response": response,
        "full_sequence_ids": sequence.detach().cpu().tolist(),
        "full_text": full_text,
        "record_step_count": len(step_dirs),
        "record_steps": step_dirs,
        "record_file_count": file_count,
        "generation_score_shapes": score_shapes,
        "dtype": args.dtype,
        "decode_mode": f"greedy do_sample=False max_new_tokens={args.max_new_tokens} use_cache=True",
    }
    (record_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
