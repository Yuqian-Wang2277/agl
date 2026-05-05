"""
MetaICL SFT training with TRL SFTTrainer (TRL >= 1.0).

Loss is computed only on the final answer (after the last "Output:" token span).
Each sample is pre-tokenised and tagged with a completion_mask
(0 = few-shot context / prompt, 1 = test-query answer).
trl.trainer.sft_trainer.DataCollatorForLanguageModeling then sets labels=-100
wherever completion_mask==0, so gradients flow only through the answer tokens.

Usage:
    python train_metaicl.py --config configs/train_qwen3_4b.yaml
    MAX_STEPS=500 python train_metaicl.py --config configs/train_qwen3_4b.yaml

Environment: conda activate agl  (pip install "trl>=1.0" transformers datasets)
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
    }
    for env_key, (cfg_key, cast) in env_map.items():
        if env_key in os.environ:
            cfg[cfg_key] = cast(os.environ[env_key])
    return cfg


# ---------------------------------------------------------------------------
# Tokenisation with completion_mask
# ---------------------------------------------------------------------------

def _tokenize_with_mask(example: dict, tokenizer, max_length: int) -> dict:
    """Tokenise one MetaICL sample and build a completion_mask.

    completion_mask[i] == 1  — token is part of the test-query answer (loss here)
    completion_mask[i] == 0  — token is few-shot context or prompt (loss masked)

    The split point is the LAST occurrence of "\\nOutput: " in the text, which
    exactly mirrors the old DataCollatorForCompletionOnlyLM behaviour with
    response_template="\\nOutput:".
    """
    text = example["text"]

    # Character offset where the answer begins
    sep = "\nOutput: "
    last_sep_pos = text.rfind(sep)
    completion_start_char = (last_sep_pos + len(sep)) if last_sep_pos != -1 else len(text)

    # Fast tokenizers support offset_mapping (precise char-to-token alignment)
    if tokenizer.is_fast:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]

        completion_mask = []
        for start, end in offsets:
            # Special tokens (BOS/EOS added by the tokenizer) have offset (0, 0)
            if start == 0 and end == 0:
                completion_mask.append(0)
            else:
                completion_mask.append(1 if start >= completion_start_char else 0)

        # The trailing EOS (if any) should be trained so the model learns to stop
        if offsets and offsets[-1] == (0, 0) and any(m == 1 for m in completion_mask[:-1]):
            completion_mask[-1] = 1
    else:
        # Slow tokenizer fallback: compare token-count of prompt vs full text
        enc = tokenizer(text, truncation=True, max_length=max_length)
        input_ids = enc["input_ids"]
        prompt_ids = tokenizer(
            text[:completion_start_char], truncation=True, max_length=max_length
        )["input_ids"]
        n_prompt = min(len(prompt_ids), len(input_ids))
        completion_mask = [0] * n_prompt + [1] * (len(input_ids) - n_prompt)

    return {"input_ids": input_ids, "completion_mask": completion_mask}


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
    from trl import SFTTrainer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    from data_formatter import DEFAULT_DATA_DIR, build_metaicl_dataset

    data_dir = Path(cfg.get("data_dir") or DEFAULT_DATA_DIR)
    k = int(cfg.get("k_shot", 4))
    model_path = cfg["model_name_or_path"]
    output_dir = cfg["output_dir"]
    max_seq_length = cfg.get("max_seq_length", 2048)
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

    logger.info("Loading tokenizer from %s...", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Pre-tokenise: each sample becomes {"input_ids": [...], "completion_mask": [...]}
    # This runs on every rank independently; HuggingFace datasets caches the result.
    logger.info("Tokenising datasets with completion_mask (max_length=%d)...", max_seq_length)
    fn_kwargs = {"tokenizer": tokenizer, "max_length": max_seq_length}
    train_dataset = (
        Dataset.from_list(train_samples)
        .map(_tokenize_with_mask, fn_kwargs=fn_kwargs,
             remove_columns=["text"], desc="Tokenising train")
    )
    val_dataset = (
        Dataset.from_list(val_samples)
        .map(_tokenize_with_mask, fn_kwargs=fn_kwargs,
             remove_columns=["text"], desc="Tokenising val")
    )

    logger.info("Loading model (bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # TRL 1.x collator reads completion_mask from each sample and sets
    # labels=-100 for positions where completion_mask==0.
    collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id,
        completion_only_loss=True,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=cfg.get("max_steps", 10000),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16),
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
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        ddp_find_unused_parameters=False,
        dataloader_num_workers=cfg.get("dataloader_num_workers", 4),
        report_to=cfg.get("report_to", "none"),
        run_name=cfg.get("run_name", None),
        seed=cfg.get("seed", 42),
    )

    # Dataset is pre-tokenised (has "input_ids"), so SFTTrainer skips its own
    # tokenisation step.  processing_class is still required for saving.
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=tokenizer,
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
