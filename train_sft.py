"""
train_world_model_qlora.py
==========================
4-bit QLoRA Supervised Fine-Tuning (SFT) for World Models.

Features:
- 4-bit Quantization (NF4) via BitsAndBytes.
- W&B Logging + HuggingFace Hub Authentication.
- Dynamic GPU utilization (Auto-batch size).
- Qwen/Llama compatible target modules.

References:
- TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer
- PEFT/LoRA: https://huggingface.co/docs/peft/index
- Weights & Biases: https://docs.wandb.ai/guides/integrations/huggingface

Usage:
python train_world_model_qlora.py \
  --wandb_key "your_wandb_api_key" \
  --hf_token "your_hf_write_token" \
  --train_csv "processed_data/training_pairs.csv" \
  --model_id "Qwen/Qwen2.5-7B-Instruct"
"""

import argparse
import os
import sys
import torch
import pandas as pd
import wandb
from datasets import Dataset
from huggingface_hub import login

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ---------- Configuration & Prompts ----------

# The system prompt enforces the JSON world model behavior
SYSTEM_PROMPT = (
    "You are a Causal World Model Simulator. "
    "Given a starting State T and a context of events occurring over the next 2 weeks, "
    "predict the resulting State T+1 in strict JSON format."
)

def formatting_prompts_func(examples):
    """
    Formats the CSV data into the Qwen/ChatML style prompt.
    """
    output_texts = []
    
    inputs = examples['input_state']
    contexts = examples['future_context']
    targets = examples['target_state']
    
    for inp, ctx, tgt in zip(inputs, contexts, targets):
        # We format this to look like a standard chat interaction
        # Note: Qwen handles standard text prompts well, but we make explicit sections
        text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Current State (T): \n{inp}\n\n"
            f"Events occurring T to T+2 Weeks:\n{ctx}\n\n"
            f"Task: Generate State T+1.<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"{tgt}<|im_end|>"
        )
        output_texts.append(text)
        
    return output_texts

def main():
    parser = argparse.ArgumentParser(description="4-bit QLoRA Training for World Models")

    # --- Mandatory Auth ---
    parser.add_argument("--wandb_key", type=str, required=True, help="Weights & Biases API Key")
    parser.add_argument("--hf_token", type=str, required=True, help="HuggingFace Write Token")

    # --- Data & Model ---
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV (input_state, future_context, target_state)")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="HF Model ID")
    parser.add_argument("--output_dir", type=str, default="./qwen_world_model_adapter", help="Output directory for adapters")
    
    # --- Hyperparameters ---
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=4096)

    args = parser.parse_args()

    # 1. Authentication
    print(f"[AUTH] Logging into HuggingFace Hub...")
    login(token=args.hf_token)

    print(f"[AUTH] Logging into Weights & Biases...")
    wandb.login(key=args.wandb_key)
    os.environ["WANDB_PROJECT"] = "macro-world-model"

    # 2. Load Dataset
    print(f"[DATA] Loading {args.train_csv}...")
    df = pd.read_csv(args.train_csv)
    dataset = Dataset.from_pandas(df)

    # 3. Hardware Check & Quantization Config
    # Check for BFloat16 support (Ampere/Hopper GPUs)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"[HARDWARE] BFloat16 Supported: {use_bf16}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 4. Load Model (Quantized)
    print(f"[MODEL] Loading {args.model_id} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2" if use_bf16 else "eager",
        trust_remote_code=True 
    )
    model.config.use_cache = False # Required for gradient checkpointing
    model = prepare_model_for_kbit_training(model)

    # 5. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Qwen usually treats eos as pad in SFT

    # 6. LoRA Configuration
    # We target all linear layers to maximize world-modeling capability
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
    )

    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=2,     # Small start, auto-finder will increase if possible
        gradient_accumulation_steps=4,     # Effective batch size = 2 * 4 = 8
        optim="paged_adamw_8bit",          # Saves VRAM
        logging_steps=5,
        save_strategy="epoch",
        learning_rate=args.lr,
        fp16=not use_bf16,
        bf16=use_bf16,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",      # Keeps learning rate steady for short finetunes
        auto_find_batch_size=True,         # Dynamic batch sizing to prevent OOM
        report_to="wandb",                 # Log metrics to W&B
        gradient_checkpointing=True,       # Trades compute for memory
    )

    # 8. Data Collator (Train on Responses Only)
    # Ensure we only calculate loss on the "Output" JSON, not the "Input" Prompt
    # For Qwen, the response starts after <|im_start|>assistant
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # 9. Trainer Initialization
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        max_seq_length=args.max_seq_length,
    )

    # 10. Train
    print("[TRAIN] Starting QLoRA Training...")
    trainer.train()

    # 11. Save
    print(f"[SAVE] Saving LoRA adapters to {args.output_dir}")
    trainer.save_model(args.output_dir)
    
    print("[DONE] Training complete. Use inference_vllm.py to run this model.")

if __name__ == "__main__":
    main()
