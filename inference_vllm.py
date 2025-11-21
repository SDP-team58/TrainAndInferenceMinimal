"""
inference_vllm.py
=================
High-performance inference for Macro World Models using vLLM.

Features:
- Auto-detects GPU count for Tensor Parallelism.
- Loads local LoRA adapters merged with base models or full models.
- Enforces strict JSON output generation via prompt structure.

Usage:
  python inference_vllm.py --model_path ./my_trained_model --input_file inputs.jsonl --output_file predictions.jsonl
"""

import argparse
import json
import torch
import os
from typing import List, Dict
from vllm import LLM, SamplingParams

# ---------- Prompt Formatting ----------
# Must match the format used during training exactly
SYSTEM_PROMPT = (
    "You are a Causal World Model Simulator. "
    "Given a starting State T and a context of events occurring over the next 2 weeks, "
    "predict the resulting State T+1 in strict JSON format."
)

def format_prompt(input_json: Dict) -> str:
    """
    Constructs the prompt from the Input State JSON.
    Assumes input_json contains 'state' (T) and 'future_context' or 'incoming_shocks'.
    """
    state_t = json.dumps(input_json.get("state", {}), indent=2)
    # In inference, we might pass the 'future news' as part of the input object
    # labeled as 'future_context' or 'shocks'
    context = input_json.get("future_context", "No specific external news provided.")
    
    prompt = (
        f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n"
        f"[USER]\n"
        f"Current State (T): \n{state_t}\n\n"
        f"Events occurring T to T+2 Weeks:\n{context}\n\n"
        f"Task: Generate State T+1.\n\n"
        f"[ASSISTANT]\n"
    )
    return prompt

# ---------- Main Inference Logic ----------

def get_gpu_count():
    return torch.cuda.device_count()

def main():
    parser = argparse.ArgumentParser(description="vLLM Inference for World Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to local model or HF repo")
    parser.add_argument("--input_file", type=str, required=True, help="JSONL file containing input states")
    parser.add_argument("--output_file", type=str, required=True, help="Output path for predictions")
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0 for deterministic, higher for rollout diversity")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens for the JSON output")
    parser.add_argument("--quantization", type=str, default=None, help="e.g., 'awq' or 'gptq' if using quantized weights")
    
    args = parser.parse_args()

    # 1. Dynamic Hardware Configuration
    num_gpus = get_gpu_count()
    print(f"[INFO] Detected {num_gpus} NVIDIA GPUs. Setting Tensor Parallelism = {num_gpus}")

    # 2. Initialize vLLM
    # trust_remote_code is often needed for newer architectures
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=num_gpus,
        quantization=args.quantization,
        trust_remote_code=True,
        # vLLM automatically manages memory, but you can tune gpu_memory_utilization if needed
        gpu_memory_utilization=0.90 
    )

    # 3. Prepare Inputs
    prompts = []
    raw_data = []
    
    print(f"[INFO] Loading inputs from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            raw_data.append(data)
            prompts.append(format_prompt(data))

    # 4. Set Sampling Parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=["[USER]", "<|endoftext|>"], # Stop tokens to prevent hallucinations continuing
    )

    # 5. Run Inference (Batch)
    print(f"[INFO] Running inference on {len(prompts)} samples...")
    outputs = llm.generate(prompts, sampling_params)

    # 6. Save Results
    print(f"[INFO] Saving results to {args.output_file}...")
    with open(args.output_file, 'w') as f_out:
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            
            # Attempt to parse JSON to ensure validity (optional but recommended logging)
            try:
                prediction_json = json.loads(generated_text)
                valid_json = True
            except json.JSONDecodeError:
                valid_json = False
            
            result_record = {
                "input_data": raw_data[i],
                "prediction_text": generated_text,
                "valid_json": valid_json
            }
            f_out.write(json.dumps(result_record) + "\n")

    print("[SUCCESS] Inference complete.")

if __name__ == "__main__":
    main()
