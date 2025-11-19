import argparse
import json
import random
from itertools import islice
from datasets import load_dataset, Dataset
from google.cloud import storage
from tqdm import tqdm

# --- Configuration ---
DATASET_NAME = "XenArcAI/MathX-5M"
NUM_EXAMPLES_TO_TAKE = 10000  # Same as your notebook
VAL_SPLIT_RATIO = 0.1  # 10% for validation
SYSTEM_PROMPT = "You are an expert mathematics tutor. Solve problems step-by-step, showing your reasoning clearly."
# ---------------------


def format_example_for_vertex(example):
    """
    Formats a single example into the Vertex AI
    {"prompt": ..., "completion": ...} schema.
    """
    if "question" in example:
        problem = example["question"]
    else:
        problem = example["problem"]

    # The 'prompt' field contains the full context *before* the model's turn
    prompt_text = f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{problem}<end_of_turn>\n<start_of_turn>model\n"

    # The 'completion' field is *only* what the model should say
    completion_text = f"{example['generated_solution']}<end_of_turn>"

    return {
        "prompt": prompt_text,
        "completion": completion_text
    }


def upload_to_gcs(bucket_name, gcs_path, local_file_path):
    """Uploads a local file to GCS."""
    print(f"Uploading {local_file_path} to gs://{bucket_name}/{gcs_path}...")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_file_path)
        print("Upload complete.")
        return f"gs://{bucket_name}/{gcs_path}"
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        print("Please ensure your GCS bucket exists and you have 'Storage Object Admin' permissions.")
        return None


def main(bucket_name, gcs_prefix):
    print(f"Loading streaming dataset: {DATASET_NAME}...")
    streamed_dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    print(f"Taking first {NUM_EXAMPLES_TO_TAKE} examples...")
    subset = list(islice(streamed_dataset, NUM_EXAMPLES_TO_TAKE))
    dataset = Dataset.from_list(subset)

    print("Formatting dataset for Vertex AI...")
    formatted_data = [format_example_for_vertex(ex) for ex in tqdm(dataset)]

    # Shuffle and split the data
    random.shuffle(formatted_data)
    split_index = int(len(formatted_data) * (1 - VAL_SPLIT_RATIO))
    train_data = formatted_data[:split_index]
    val_data = formatted_data[split_index:]

    print(f"Total examples: {len(formatted_data)}")
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")

    # Write to local JSONL files
    train_file_local = "mathx_train.jsonl"
    val_file_local = "mathx_validation.jsonl"

    with open(train_file_local, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(val_file_local, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    print("\nLocal files created. Now uploading to GCS...")

    # Upload to GCS
    train_uri = upload_to_gcs(bucket_name, f"{gcs_prefix}/train.jsonl", train_file_local)
    val_uri = upload_to_gcs(bucket_name, f"{gcs_prefix}/validation.jsonl", val_file_local)

    if train_uri and val_uri:
        print("\n--- 🚀 Data Preparation Complete! ---")
        print("You can now use these URIs to launch your training jobs:")
        print(f"Train URI: {train_uri}")
        print(f"Validation URI: {val_uri}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bucket",
        type=str,
        required=True,
        help="Your GCS bucket name (e.g., 'my-model-tuning-bucket').",
    )
    parser.add_argument(
        "--gcs-prefix",
        type=str,
        default="mathx-dataset",
        help="The folder path inside your GCS bucket to store the data.",
    )
    args = parser.parse_args()
    main(args.bucket, args.gcs_prefix)
