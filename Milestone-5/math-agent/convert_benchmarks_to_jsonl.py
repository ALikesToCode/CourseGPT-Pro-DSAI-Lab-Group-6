import json
import os

# Input directory containing benchmark folders and .jsonl files
INPUT_DIR = "benchmarks_dataset"
# Output combined JSONL file
OUTPUT_FILE = "combined_benchmarks.jsonl"

# System message used for all examples
SYSTEM_PROMPT = (
    "You are a helpful math assistant. Your primary goal is to accurately solve mathematical "
    "problems and provide clear, step-by-step explanations for your reasoning.\n\n"
    "When responding to a math query:\n"
    "1.  *Analyze:* Carefully read the problem to understand exactly what is being asked.\n"
    "2.  *Show Your Work:* Break down the solution into logical, easy-to-follow steps.\n"
    "3.  *Use Formatting:* Utilize LaTeX for all complex mathematical equations, formulas, and variables "
    "to ensure clarity and correct rendering (e.g., $$E=mc^2$$).\n"
    "4.  *Final Answer:* Clearly state the final answer at the end of your explanation."
)


def convert_to_new_format(entry):
    """
    Convert one benchmark entry to the new JSONL format.
    Handles both single-choice and closed questions.
    """
    question = entry.get("question", "").strip()
    options = entry.get("options")

    if not question:
        return None  # Skip invalid entries

    # Build the user's message content
    if options and isinstance(options, list):
        content = f"{question}\n\nOptions:\n"
        for i, opt in enumerate(options):
            content += f"{chr(65+i)}. {opt}\n"
    else:
        content = question

    # New JSONL structure with temperature field
    formatted = {
        "body": {
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
        },
        "temperature": 0
    }
    return formatted


def combine_benchmarks(input_dir, output_file):
    """
    Walk through all folders and combine .jsonl files into one combined JSONL.
    """
    count = 0
    with open(output_file, "w", encoding="utf-8") as out:
        for root, _, files in os.walk(input_dir):
            for filename in files:
                if not filename.endswith(".jsonl"):
                    continue

                filepath = os.path.join(root, filename)
                print(f"📘 Processing {filepath}")

                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            print(f"⚠️ Skipping invalid JSON in {filename}")
                            continue

                        formatted = convert_to_new_format(entry)
                        if formatted:
                            out.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                            count += 1

    if count == 0:
        print("❌ No valid data found. Check if your JSONL files contain 'question' keys.")
    else:
        print(f"✅ Successfully wrote {count} entries to {output_file}")


if __name__ == "__main__":
    combine_benchmarks(INPUT_DIR, OUTPUT_FILE)
