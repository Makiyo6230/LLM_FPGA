"""Explicit Qwen2.5-0.5B prefill/decode inference loop.

Unlike run_full_generate_trace.py, this script does not call model.generate().
It shows the exact autoregressive flow that FPGA runtime needs to reproduce:

  prompt -> tokenizer/chat template -> prefill forward -> argmax next token
  -> decode forward with past_key_values -> argmax -> repeat

When used with the instrumented Transformers source and QWEN2_FPGA_RECORD=1,
each model forward call creates one trace step:

  step_0000 = prefill full prompt
  step_0001 = decode generated token 01
  ...
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
    parser = argparse.ArgumentParser(description="Run explicit Qwen2.5 prefill/decode loop.")
    parser.add_argument("--model-path", default="/data3/models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--record-dir", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--save-scores", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def save_score(record_dir: Path, step_index: int, logits: torch.Tensor) -> list[int]:
    score_dir = record_dir / "generation_token_vocab_logits"
    score_dir.mkdir(parents=True, exist_ok=True)
    score = logits[:, -1, :].detach().cpu().to(torch.float32).numpy()
    if step_index == 0:
        name = "prefill_next_token_vocab_logits.npy"
    else:
        name = f"decode_step_{step_index:04d}_next_token_vocab_logits.npy"
    np.save(score_dir / name, score)
    return list(score.shape)


def main() -> None:
    args = parse_args()
    if args.record_dir:
        record_dir = Path(args.record_dir)
        record_dir.mkdir(parents=True, exist_ok=True)
        os.environ["QWEN2_FPGA_RECORD"] = "1"
        os.environ["QWEN2_FPGA_RECORD_DIR"] = str(record_dir)
    else:
        record_dir = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([chat_text], return_tensors="pt")
    input_ids = model_inputs.input_ids.to(args.device)
    attention_mask = model_inputs.attention_mask.to(args.device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation="eager",
    ).to(args.device)
    model.eval()

    generated_ids: list[int] = []
    score_shapes: list[list[int]] = []
    past_key_values = None

    with torch.no_grad():
        # Prefill: full prompt enters the model.
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids.append(int(next_token.item()))
        if args.save_scores and record_dir is not None:
            score_shapes.append(save_score(record_dir, 0, outputs.logits))

        # Decode: one token enters the model at each step, with KV cache.
        for step_index in range(1, args.max_new_tokens):
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), device=args.device, dtype=attention_mask.dtype)],
                dim=-1,
            )
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids.append(int(next_token.item()))
            if args.save_scores and record_dir is not None:
                score_shapes.append(save_score(record_dir, step_index, outputs.logits))

    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
    metadata = {
        "model_path": args.model_path,
        "record_dir": str(record_dir) if record_dir else None,
        "prompt": args.prompt,
        "messages": messages,
        "chat_text": chat_text,
        "input_token_count": int(input_ids.shape[1]),
        "input_ids": input_ids[0].detach().cpu().tolist(),
        "generated_token_count": len(generated_ids),
        "generated_ids": generated_ids,
        "generated_tokens": generated_tokens,
        "response": response,
        "score_shapes": score_shapes,
        "decode_mode": f"manual greedy argmax max_new_tokens={args.max_new_tokens} use_cache=True",
    }
    if record_dir is not None:
        (record_dir / "manual_prefill_decode_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
