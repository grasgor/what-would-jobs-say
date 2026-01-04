import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel, PeftConfig

def generate_response(args):
    print(f"Loading base model: {args.base_model}")
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # Load adapters
    if args.adapter_path:
        print(f"Loading adapters from: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    
    # Merge and unload if specified (optional, but faster inference)
    if args.merge:
        print("Merging adapters...")
        model = model.merge_and_unload()

    # Create pipeline
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        device_map="auto"
    )

    print(f"Generating response for prompt: {args.prompt}")
    result = pipe(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty
    )
    
    print("\n--- Generated Text ---\n")
    print(result[0]["generated_text"])
    print("\n----------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using the Steve Jobs model.")
    
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B", help="Base model name")
    parser.add_argument("--adapter_path", type=str, default="./llama3.2_jobs_sft", help="Path to trained adapters")
    parser.add_argument("--merge", action="store_true", help="Merge adapters into base model for inference")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top K")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top P")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty")

    args = parser.parse_args()
    generate_response(args)
