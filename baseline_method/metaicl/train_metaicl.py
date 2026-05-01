"""
MetaICL SFT training with TRL SFTTrainer.

Loss is computed only on the final answer (after the last "Output:" token span).
DataCollatorForCompletionOnlyLM scans for the LAST occurrence of response_template
in each sequence — exactly the test-query answer, not the few-shot answers.

Usage:
    python train_metaicl.py --config configs/train_qwen3_4b.yaml
    MAX_STEPS=500 python train_metaicl.py --config configs/train_qwen3_4b.yaml

Environment: conda activate agl  (pip install trl if not already installed)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config(config_path: str, cli_overrides: dict) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # CLI args override YAML; skip None values
    for k, v in cli_overrides.items():
        if v is not None:
            cfg[k] = v
    # Environment variable overrides (same keys, upper-cased)
    env_map = {
        "MAX_STEPS": ("max_steps", int),
        "K_SHOT": ("k_shot", int),
        "LR": ("learning_rate", float),
        "MODEL_PATH": ("model_name_or_path", str),
        "DATA_DIR": ("data_dir", str),
        "OUTPUT_DIR": ("output_dir", str),
    }
    for env_key, (cfg_key, cast) in env_map.items():
        if env_key in os.environ:
            cfg[cfg_key] = cast(os.environ[env_key])
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MetaICL SFT training")
    parser.add_argument("--config", default="configs/train_qwen3_4b.yaml")
    parser.add_argument("--model-path", dest="model_name_or_path", default=None)
    parser.add_argument("--data-dir", dest="data_dir", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=None)
    parser.add_argument("--k-shot", dest="k_shot", type=int, default=None)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=None)
    parser.add_argument("--batch-size", dest="per_device_train_batch_size", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_config(
        args.config,
        {
            "model_name_or_path": args.model_name_or_path,
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
            "max_steps": args.max_steps,
            "k_shot": args.k_shot,
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
        },
    )

    # Late imports — keep startup fast, avoid loading torch before config is validated
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

    logger.info("Building dataset from %s (k=%d, samples_per_task=%d)...",
                data_dir, k, cfg.get("samples_per_task", 2000))
    train_samples, val_samples = build_metaicl_dataset(
        data_dir=data_dir,
        k=k,
        samples_per_task=cfg.get("samples_per_task", 2000),
        val_ratio=cfg.get("val_ratio", 0.05),
        seed=cfg.get("data_seed", 42),
    )
    logger.info("Train: %d  Val: %d", len(train_samples), len(val_samples))

    train_dataset = Dataset.from_list(train_samples)
    val_dataset = Dataset.from_list(val_samples)

    logger.info("Loading tokenizer from %s...", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Loading model (bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # The leading \n in response_template means we match "\nOutput:" which
    # appears before EVERY Output: in the sequence.  TRL's collator keeps the
    # LAST matching position, so loss is computed only on the test-query answer.
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

    logger.info("Starting training (max_steps=%d)...", cfg.get("max_steps", 10000))
    trainer.train()

    best_dir = os.path.join(output_dir, "best")
    logger.info("Saving best checkpoint → %s", best_dir)
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
