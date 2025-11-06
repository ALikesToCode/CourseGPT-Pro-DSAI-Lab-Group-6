import os
from dataclasses import dataclass
from typing import Dict

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class MergeConfig:
    name: str
    base_model: str
    adapter_repo: str
    output_dir: str
    push_repo: str

    @classmethod
    def from_defaults(cls, name: str, defaults: Dict[str, str]) -> "MergeConfig":
        upper = name.upper()
        base = os.environ.get(f"BASE_MODEL_{upper}", defaults["base_model"])
        adapter = os.environ.get(f"ADAPTER_REPO_{upper}", defaults["adapter_repo"])
        output = os.environ.get(f"OUTPUT_DIR_{upper}", defaults["output_dir"])
        push = os.environ.get(f"PUSH_REPO_{upper}", defaults["push_repo"])
        return cls(name=name, base_model=base, adapter_repo=adapter, output_dir=output, push_repo=push)


DEFAULT_TARGETS: Dict[str, Dict[str, str]] = {
    "qwen": {
        "base_model": "Qwen/Qwen3-32B",
        "adapter_repo": "CourseGPT-Pro-DSAI-Lab-Group-6/router-qwen3-32b-peft",
        "output_dir": "./router-qwen3-32b-merged",
        "push_repo": "Alovestocode/router-qwen3-32b-merged",
    },
    "llama": {
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "adapter_repo": "CourseGPT-Pro-DSAI-Lab-Group-6/router-llama31-peft",
        "output_dir": "./router-llama31-merged",
        "push_repo": "Alovestocode/router-llama31-merged",
    },
    "gemma": {
        "base_model": "google/gemma-3-27b-it",
        "adapter_repo": "CourseGPT-Pro-DSAI-Lab-Group-6/router-gemma3-peft",
        "output_dir": "./router-gemma3-merged",
        "push_repo": "Alovestocode/router-gemma3-merged",
    },
}


PUSH_TO_HUB = os.environ.get("PUSH_TO_HUB", "true").lower() == "true"


def merge_single(cfg: MergeConfig) -> None:
    print(f"\n========== Merging target: {cfg.name} ==========")
    print(f"Base model   : {cfg.base_model}")
    print(f"Adapter repo : {cfg.adapter_repo}")
    print(f"Output dir   : {cfg.output_dir}")
    print(f"Push repo    : {cfg.push_repo if PUSH_TO_HUB else '(push disabled)'}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=False)

    print(">> Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token

    print(">> Loading LoRA adapter...")
    peft_model = PeftModel.from_pretrained(base_model, cfg.adapter_repo, is_trainable=False)

    print(">> Merging LoRA into base weights...")
    merged_model = peft_model.merge_and_unload()

    print(f">> Saving merged checkpoint to {cfg.output_dir} ...")
    os.makedirs(cfg.output_dir, exist_ok=True)
    tokenizer.save_pretrained(cfg.output_dir)
    merged_model.save_pretrained(
        cfg.output_dir,
        safe_serialization=True,
        max_shard_size="10GB",
    )

    if PUSH_TO_HUB:
        print(f">> Creating (if needed) and uploading to Hugging Face: {cfg.push_repo}")
        api = HfApi()
        api.create_repo(repo_id=cfg.push_repo, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=cfg.output_dir,
            repo_id=cfg.push_repo,
            repo_type="model",
            commit_message=f"Add merged router checkpoint ({cfg.name})",
        )

    print(f">> Finished target: {cfg.name}")


def main() -> None:
    targets_env = os.environ.get("ROUTER_TARGETS")
    if targets_env:
        targets = [t.strip().lower() for t in targets_env.split(",") if t.strip()]
    else:
        targets = ["qwen"]  # default for backwards compatibility

    missing = [t for t in targets if t not in DEFAULT_TARGETS]
    if missing:
        raise ValueError(f"Unknown targets requested: {missing}. Available: {list(DEFAULT_TARGETS.keys())}")

    for name in targets:
        cfg = MergeConfig.from_defaults(name, DEFAULT_TARGETS[name])
        merge_single(cfg)

    print("\nAll requested merges completed.")


if __name__ == "__main__":
    main()
