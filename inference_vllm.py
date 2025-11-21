"""
inference_vllm.py
=================
High-performance inference for World Models with JSON Validation & Retry Logic.

Features:
- Uses vLLM for maximum throughput.
- Raw Input: Passes user input directly to the model (assumes SFT handling).
- Validation Loop: Automatically retries generation for failed JSON outputs.

Usage:
  python inference_vllm.py \
    --model_path ./qwen_world_model_adapter \
    --input_file inputs.jsonl \
    --output_file predictions.jsonl \
    --max_retries 5
"""

import argparse
import json
import torch
import sys
from typing import List, Dict, Any
from vllm import LLM, SamplingParams

def validate_json(text: str) -> bool:
    """
    Checks if the text is valid JSON. 
    For world models, we often want to strip Markdown ```json ... ``` wrappers 
    if the model adds them.
    """
    text = text.strip()
    # Simple cleanup for common LLM markdown habits
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False

def clean_output_text(text: str) -> str:
    """Helper to extract JSON string from potential markdown wrappers."""
    original = text
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def main():
    parser = argparse.ArgumentParser(description="vLLM Inference with Retry Logic")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model or LoRA adapter")
    parser.add_argument("--input_file", type=str, required=True, help="JSONL file containing inputs")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save results")
    parser.add_argument("--input_field", type=str, default="input", help="Key in JSONL to use as prompt")
    
    # Generation Params
    parser.add_argument("--max_retries", type=int, default=3, help="How many times to retry invalid JSON")
    parser.add_argument("--temperature", type=float, default=0.7, help="Higher temp helps retries find new paths")
    parser.add_argument("--max_tokens", type=int, default=4096)
    
    args = parser.parse_args()

    # 1. Setup Hardware
    num_gpus = torch.cuda.device_count()
    print(f"[INFO] Detected {num_gpus} GPUs. Initializing vLLM...")

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        # Ensure we don't cut off JSON
        max_model_len=args.max_tokens + 1024 
    )

    # 2. Load Data
    print(f"[INFO] Loading inputs from {args.input_file}...")
    all_records = []
    with open(args.input_file, 'r') as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line))

    # Track which records are done and which need processing
    # pending_indices maps {index_in_all_records -> prompt_string}
    pending_indices = {
        i: record.get(args.input_field, "") 
        for i, record in enumerate(all_records)
    }
    
    # Storage for final results
    # final_results[i] = {"output": ..., "valid": bool, "attempts": int}
    final_results: Dict[int, Any] = {}

    # 3. Retry Loop
    attempt = 0
    while pending_indices and attempt <= args.max_retries:
        
        current_batch_ids = list(pending_indices.keys())
        current_batch_prompts = list(pending_indices.values())
        
        print(f"\n[STEP] Attempt {attempt + 1}/{args.max_retries + 1}")
        print(f"[STEP] Processing {len(current_batch_ids)} items...")

        # Dynamic sampling: increase temp slightly on retries to avoid getting stuck
        current_temp = args.temperature + (0.1 * attempt) if attempt > 0 else args.temperature
        # Cap temp at 1.0 to prevent gibberish
        current_temp = min(current_temp, 1.0) 

        sampling_params = SamplingParams(
            temperature=current_temp,
            max_tokens=args.max_tokens,
            # Stop tokens help prevent the model from rambling after closing JSON
            stop=["<|im_end|>", "<|endoftext|>"] 
        )

        # Run Inference
        outputs = llm.generate(current_batch_prompts, sampling_params)

        # Process Results
        success_count = 0
        
        for i, output_obj in enumerate(outputs):
            original_idx = current_batch_ids[i]
            generated_text = output_obj.outputs[0].text
            
            is_valid = validate_json(generated_text)
            
            if is_valid:
                # Success: Store and remove from pending
                final_results[original_idx] = {
                    "prediction": clean_output_text(generated_text),
                    "raw_text": generated_text,
                    "valid_json": True,
                    "attempts": attempt + 1
                }
                del pending_indices[original_idx]
                success_count += 1
            else:
                # Failure: Keep in pending if we have retries left
                # If this was the last attempt, store the failure
                if attempt == args.max_retries:
                    final_results[original_idx] = {
                        "prediction": generated_text, # Store raw invalid text
                        "raw_text": generated_text,
                        "valid_json": False,
                        "attempts": attempt + 1
                    }
                    del pending_indices[original_idx]

        print(f"[STEP] Batch finished. {success_count} valid JSON outputs. {len(pending_indices)} remaining.")
        attempt += 1

    # 4. Save Final Outputs
    print(f"\n[INFO] Saving {len(final_results)} results to {args.output_file}...")
    
    # We want to write them out in the same order as input
    with open(args.output_file, 'w') as f_out:
        for i in range(len(all_records)):
            res = final_results.get(i)
            if not res:
                # Should not happen logic-wise unless empty input
                continue
                
            output_record = {
                "input_data": all_records[i],
                "model_output": res["prediction"],
                "meta": {
                    "valid_json": res["valid_json"],
                    "attempts": res["attempts"]
                }
            }
            f_out.write(json.dumps(output_record) + "\n")

    print("[SUCCESS] Inference Complete.")

if __name__ == "__main__":
    main()
