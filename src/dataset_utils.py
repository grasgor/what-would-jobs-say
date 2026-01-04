from datasets import load_dataset, load_from_disk
import os

def preprocess_sft_function(row):
    """
    Preprocesses the dataset rows to format them for SFT.
    """
    return {
        "prompt": row["instruction"],      # simple string
        "completion": row["output"]        # simple string
    }

def load_and_prepare_sft_dataset(dataset_name="Hypersniper/Steve_Jobs_Interviews", split="train"):
    """
    Loads and pre-processes the Steve Jobs dataset for SFT.
    """
    dataset = load_dataset(dataset_name)
    
    # Apply preprocessing
    processed_dataset = dataset[split].map(
        preprocess_sft_function, 
        remove_columns=["instruction", "output"]
    )
    
    return processed_dataset

def load_and_prepare_dpo_dataset(dataset_path):
    """
    Loads and prepares the DPO dataset (multi-reject format).
    Expects a dataset with columns: prompt, completion, rejected_1, rejected_2, rejected_3.
    """
    if os.path.exists(dataset_path):
        ds = load_from_disk(dataset_path)
    else:
        # Fallback to loading from HF if it's a hub path
        ds = load_dataset(dataset_path, split="train")

    records = []
    for row in ds:
        # Handle cases where columns might be named differently or missing
        # The notebook expects prompt, completion, rejected_1, rejected_2, rejected_3
        rec = {
            "prompt": row["prompt"],
            "chosen": row.get("completion"), # 'completion' is the chosen answer
            "rejected_1": row.get("rejected_1"),
            "rejected_2": row.get("rejected_2"),
            "rejected_3": row.get("rejected_3"),
            # Store all rejects in a list for easier processing
            "rejected": [row.get("rejected_1"), row.get("rejected_2"), row.get("rejected_3")]
        }
        # Filter out examples with missing data if necessary, or keep valid ones
        if rec["chosen"] and any(rec["rejected"]):
             records.append(rec)
             
    print(f"Loaded {len(records)} DPO examples.")
    return records
