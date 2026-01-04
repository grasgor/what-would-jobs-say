import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import wandb
from dataset_utils import load_and_prepare_sft_dataset

def train(args):
    # Login to wandb if key is provided
    if args.wandb_key:
        os.environ["WANDB_API_KEY"] = args.wandb_key
        wandb.login()

    # Load Dataset
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_and_prepare_sft_dataset(args.dataset_name)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # BitsAndBytes Config (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )

    # Load Model
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={'': torch.cuda.current_device()} if torch.cuda.is_available() else "auto"
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # LoRA Config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # SFT Config
    sft_config = SFTConfig(
        completion_only_loss=True,
        output_dir=args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=20,
        fp16=True,
        save_strategy="epoch",
        save_total_limit=3,
        max_length=args.max_seq_length,
        report_to="wandb" if args.use_wandb else "none",
    )

    # Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        peft_config=peft_config,
        formatting_func=None # dataset is already preprocessed
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save Model
    print(f"Saving model to {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model on Steve Jobs interviews.")
    
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Base model name")
    parser.add_argument("--dataset_name", type=str, default="Hypersniper/Steve_Jobs_Interviews", help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="./llama3.2_jobs_sft", help="Output directory")
    
    # Hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--max_seq_length", type=int, default=3072, help="Max sequence length")
    
    # LoRA Hyperparameters
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA r value")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha value")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")

    # WandB
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_key", type=str, default=None, help="WandB API key")

    args = parser.parse_args()
    train(args)
