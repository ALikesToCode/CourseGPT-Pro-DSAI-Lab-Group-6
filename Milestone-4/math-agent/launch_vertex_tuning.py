import argparse
import os
import vertexai
from vertexai.preview.tuning import sft


def main(args):
    # Set defaults from environment if not provided
    PROJECT_ID = args.project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    REGION = args.region or os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

    if not PROJECT_ID:
        raise ValueError("Error: GOOGLE_CLOUD_PROJECT environment variable or --project-id flag must be set.")

    print(f"Initializing Vertex AI for project '{PROJECT_ID}' in region '{REGION}'...")
    vertexai.init(project=PROJECT_ID, location=REGION)

    print(f"\n--- Starting Vertex AI SFT Job ---")
    print(f"Display Name: {args.display_name}")
    print(f"Base Model: {args.base_model}")
    print(f"Training Data: {args.train_uri}")
    print(f"Validation Data: {args.validation_uri}")
    print(f"Output Path: {args.output_uri}")
    print(f"Tuning Mode: {args.tuning_mode}")
    print(f"Adapter Size: {args.adapter_size}")
    print(f"Epochs: {args.epochs}")
    print("------------------------------------")

    # This is the function that calls the cheap, managed service
    job = sft.preview_train(
        display_name=args.display_name,
        source_model=args.base_model,
        train_dataset=args.train_uri,
        validation_dataset=args.validation_uri,
        # This is where the final LoRA adapter will be saved
        output_dir=args.output_uri,
        # PEFT is the cheap, fast LoRA method
        tuning_mode=args.tuning_mode,
        adapter_size=args.adapter_size,
        epochs=args.epochs
    )

    print("\n✅ Job submitted successfully!")
    print(f"Vertex Job Name: {job.resource_name}")
    print("Monitor progress in the Vertex AI 'Tuning' dashboard in your Google Cloud console.")

    if args.wait:
        print("Waiting for job to complete... (This can take several hours)")
        job.wait()
        print("\n🎉 Job finished!")
        print(job.result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch a Vertex AI SFT Job.")

    # --- Required Arguments ---
    parser.add_argument("--base-model", type=str, required=True, help="The base model identifier (e.g., 'meta/llama3_1@llama-3.1-8b-instruct').")
    parser.add_argument("--train-uri", type=str, required=True, help="GCS URI to the train.jsonl file.")
    parser.add_argument("--validation-uri", type=str, required=True, help="GCS URI to the validation.jsonl file.")
    parser.add_argument("--output-uri", type=str, required=True, help="GCS URI to save the final adapter (e.g., 'gs://my-bucket/adapters/gemma-4b-lora').")
    parser.add_argument("--display-name", type=str, required=True, help="A human-readable name for the job in the Vertex dashboard.")

    # --- Optional Arguments ---
    parser.add_argument("--tuning-mode", type=str, default="PEFT_ADAPTER", help="Tuning mode: 'PEFT_ADAPTER' (LoRA) or 'FULL' (Full fine-tune).")
    parser.add_argument("--adapter-size", type=int, default=16, help="LoRA adapter rank (e.g., 4, 8, 16).")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--project-id", type=str, default=None, help="Your Google Cloud Project ID. Defaults to env var GOOGLE_CLOUD_PROJECT.")
    parser.add_argument("--region", type=str, default="us-central1", help="Google Cloud region (e.g., 'us-central1').")
    parser.add_argument("--wait", action="store_true", help="Wait for the job to complete instead of exiting.")

    args = parser.parse_args()
    main(args)
