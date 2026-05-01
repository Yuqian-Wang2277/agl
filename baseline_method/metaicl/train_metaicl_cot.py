"""
MetaICL-CoT SFT training.

Adds a "Think:" field before the final "Output:" in every training sample.
Loss is still computed only on the content after the last "Output:".

Think label source (configure via THINK_SOURCE env var or --think-source):
  empty   (default) — Think field is left blank; model learns the format
                       but is not supervised on reasoning content.
                       At inference the model generates freely after "Think:".
  file    — Load pre-generated reasoning from a JSONL file (--think-file).
            Each line: {"input": str, "think": str}  keyed by exact input text.

Usage:
    python train_metaicl_cot.py --config configs/train_qwen3_4b_cot.yaml
    THINK_SOURCE=empty python train_metaicl_cot.py --config configs/train_qwen3_4b_cot.yaml

Environment: conda activate agl  (pip install trl if not already installed)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))


def _load_config(config_path: str, cli_overrides: dict) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for k, v in cli_overrides.items():
        if v is not None:
            cfg[k] = v
    env_map = {
        "MAX_STEPS": ("max_steps", int),
        "K_SHOT": ("k_shot", int),
        "LR": ("learning_rate", float),
        "MODEL_PATH": ("model_name_or_path", str),
        "DATA_DIR": ("data_dir", str),
        "OUTPUT_DIR": ("output_dir", str),
        "THINK_SOURCE": ("think_source", str),
    }
    for env_key, (cfg_key, cast) in env_map.items():
        if env_key in os.environ:
            cfg[cfg_key] = cast(os.environ[env_key])
    return cfg


def _build_think_lookup(think_file: str) -> dict:
    """Load pre-generated Think labels keyed by exact input text."""
    lookup = {}
    with open(think_file, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            lookup[obj["input"]] = obj.get("think", "")
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(description="MetaICL-CoT SFT training")
    parser.add_argument("--config", default="configs/train_qwen3_4b_cot.yaml")
    parser.add_argument("--model-path", dest="model_name_or_path", default=None)
    parser.add_argument("--data-dir", dest="data_dir", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=None)
    parser.add_argument("--k-shot", dest="k_shot", type=int, default=None)
    parser.add_argument("--think-source", dest="think_source", default=None,
                        choices=["empty", "file"])
    parser.add_argument("--think-file", dest="think_file", default=None,
                        help="JSONL file with pre-generated Think labels (think_source=file)")
    args = parser.parse_args()

    cfg = _load_config(args.config, {
        "model_name_or_path": args.model_name_or_path,
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "max_steps": args.max_steps,
        "k_shot": args.k_shot,
        "think_source": args.think_source,
    })

    think_source = cfg.get("think_source", "empty")
    think_lookup: dict = {}
    if think_source == "file":
        think_file = args.think_file or cfg.get("think_file")
        if not think_file:
            raise ValueError("think_source=file requires --think-file or think_file in config")
        think_lookup = _build_think_lookup(think_file)
        logger.info("Loaded %d think labels from %s", len(think_lookup), think_file)

    def reasoning_fn(shot_examples, test_example) -> str:
        if think_source == "file":
            return think_lookup.get(test_example.get("input", ""), "")
        return ""  # empty: model learns format but not content

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

    from data_formatter import DEFAULT_DATA_DIR, build_metaicl_dataset

    data_dir = Path(cfg.get("data_dir") or DEFAULT_DATA_DIR)
    k = int(cfg.get("k_shot", 4))
    model_path = cfg["model_name_or_path"]
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Building CoT dataset (think_source=%s, k=%d)...", think_source, k)
    train_samples, val_samples = build_metaicl_dataset(
        data_dir=data_dir,
        k=k,
        samples_per_task=cfg.get("samples_per_task", 2000),
        val_ratio=cfg.get("val_ratio", 0.05),
        seed=cfg.get("data_seed", 42),
        cot_reasoning_fn=reasoning_fn,
    )
    logger.info("Train: %d  Val: %d", len(train_samples), len(val_samples))

    train_dataset = Dataset.from_list(train_samples)
    val_dataset = Dataset.from_list(val_samples)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    # CoT format ends with "Think: {text}\nOutput: {answer}"
    # response_template "\nOutput:" still correctly finds the LAST Output:
    response_template = "\nOutput:"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=cfg.get("max_steps", 10000),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=cfg.get("learning_rate", 1e-5),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=cfg.get("warmup_steps", 100),
        logging_steps=cfg.get("logging_steps", 50),
        eval_strategy="steps",
        eval_steps=cfg.get("eval_steps", 500),
        save_strategy="steps",
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=cfg.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        dataloader_num_workers=cfg.get("dataloader_num_workers", 4),
        report_to=cfg.get("report_to", "none"),
        seed=cfg.get("seed", 42),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        max_seq_length=cfg.get("max_seq_length", 2048),
        dataset_text_field="text",
    )

    logger.info("Starting CoT training (max_steps=%d)...", cfg.get("max_steps", 10000))
    trainer.train()

    best_dir = os.path.join(output_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    logger.info("Saved best checkpoint → %s", best_dir)


if __name__ == "__main__":
    main()
