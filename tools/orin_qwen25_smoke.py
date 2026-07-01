import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen2.5-0.5B-Instruct on Jetson Orin.")
    parser.add_argument(
        "--model-path",
        default="/home/nvidia/models/Qwen2.5-0.5B-Instruct",
        help="Local Qwen model directory on Orin.",
    )
    parser.add_argument(
        "--prompt",
        default="你好，请用一句话介绍 Qwen2.5。",
        help="User prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum number of generated tokens.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"model_path: {args.model_path}")
    print(f"prompt: {args.prompt}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check Jetson PyTorch installation.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to("cuda")
    model.eval()
    load_seconds = time.time() - load_start

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to("cuda")

    generate_start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    generate_seconds = time.time() - generate_start

    input_tokens = int(inputs.input_ids.shape[1])
    new_tokens = int(output_ids.shape[1] - input_tokens)
    response = tokenizer.decode(output_ids[0][input_tokens:], skip_special_tokens=True)

    print(f"load_seconds: {load_seconds:.2f}")
    print(f"input_tokens: {input_tokens}")
    print(f"new_tokens: {new_tokens}")
    print(f"generate_seconds: {generate_seconds:.2f}")
    print(f"tokens_per_second: {new_tokens / generate_seconds:.2f}")
    print(f"response: {response}")


if __name__ == "__main__":
    main()
