"""Phase II main training loop — Actor-Judge Co-Evolution.

Launch with:
    accelerate launch --config_file accelerate_fsdp.yaml train.py

Architecture (all four Review rounds incorporated):

M1  vLLM runs ONLY on Rank 0.  Other ranks block at wait_for_everyone().
    Generated experiences are broadcast via broadcast_object_list.

P2  Ray remote is declared with num_gpus=0 so Ray's scheduler never fights
    PyTorch FSDP for GPU ownership.  vLLM's tensor_parallel_size handles GPUs.

P3  gpu_memory_utilization ≤ 0.75 (PyTorch CUDA Context pins ~1-1.5 GB/GPU).

P4  Experience objects carry ONLY pure-Python scalars — no GPU Tensors —
    so pickle/broadcast never triggers an NCCL hang.

P5  ref_model is wrapped with accelerator.prepare() to distribute (shard) it
    across ranks rather than replicating an 8 GB model on every rank.

M6  <|judge|> embedding is warm-started from <|im_end|> after resize.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Optional

import ray
import torch
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from actor_trainer import ActorTrainer
from buffer import UCBBuffer
from config import ActorJudgeConfig
from data_loader import (
    ActorJudgeDataset,
    collate_rollout_samples,
    load_rollout_dataset,
    load_val_dataset,
)
from judge_model import JudgeModel, warmstart_judge_token_embedding
from judge_trainer import JudgeTrainer
from rollout_engine import RolloutEngine, VLLMActor

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# ---------------------------------------------------------------------------
# Broadcast helper (Rank 0 → all ranks)
# ---------------------------------------------------------------------------

def broadcast_object_list_from_rank0(obj, accelerator: Accelerator):
    """Broadcast a Python object from Rank 0 to all other ranks.

    Uses torch.distributed.broadcast_object_list internally.
    P4: obj must contain NO GPU Tensors — only pure Python scalars.
    """
    container = [obj]
    torch.distributed.broadcast_object_list(container, src=0)
    return container[0]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_pass_at_1(
    actor: torch.nn.Module,
    tokenizer,
    val_dataset: ActorJudgeDataset,
    cfg: ActorJudgeConfig,
    accelerator: Accelerator,
    split_name: str,
) -> float:
    """Greedy-decode validation: measure Pass@1 on *val_dataset*."""
    from prompts import apply_chat_template, build_strategy_prompt, build_answer_prompt
    from env import evaluate

    actor.eval()
    correct = 0
    total   = 0

    for sample in val_dataset:
        messages1 = build_strategy_prompt(sample.fewshot_examples)
        prompt1   = apply_chat_template(tokenizer, messages1)
        enc1      = tokenizer(prompt1, return_tensors="pt").to(accelerator.device)
        with torch.no_grad():
            out1 = accelerator.unwrap_model(actor).generate(
                **enc1, max_new_tokens=512, do_sample=False
            )
        s_text = tokenizer.decode(out1[0][enc1["input_ids"].shape[1]:], skip_special_tokens=False)
        s_text = s_text + "</strategy>"

        messages2 = build_answer_prompt(s_text, sample.question)
        prompt2   = apply_chat_template(tokenizer, messages2)
        enc2      = tokenizer(prompt2, return_tensors="pt").to(accelerator.device)
        with torch.no_grad():
            out2 = accelerator.unwrap_model(actor).generate(
                **enc2, max_new_tokens=256, do_sample=False
            )
        a_text = tokenizer.decode(out2[0][enc2["input_ids"].shape[1]:], skip_special_tokens=False)
        a_text = a_text + "</answer>"

        outcome = evaluate(s_text, a_text, sample.answer_gold)
        if outcome == 1:
            correct += 1
        if outcome >= 0:
            total += 1

    actor.train()
    pass1 = correct / max(total, 1)
    logger.info("Validation [%s]: Pass@1 = %.3f (%d/%d)", split_name, pass1, correct, total)
    return pass1


# ---------------------------------------------------------------------------
# JOA: Judge-Outcome Agreement (plan §5)
# ---------------------------------------------------------------------------

def compute_joa(
    judge_model: torch.nn.Module,
    tokenizer,
    buffer,
    cfg,
    accelerator: Accelerator,
    n_samples: int = 200,
) -> float:
    """JOA = fraction(y=1 | σ(v_judge) > 0.6) measured on buffer samples.

    Plan §5: 'Judge 高分（>0.6）样本中 y=1 的比例（在验证集上每 N epoch 计算）'

    We sample `n_samples` individual experiences from the buffer (not pairwise),
    run the Judge in eval mode, and report the agreement rate.

    Returns:
        JOA value in [0, 1], or -1.0 if insufficient buffer data.
    """
    from prompts import build_judge_prompt

    all_exps = buffer.sample_individual(n_samples)
    if len(all_exps) < 10:
        logger.warning("compute_joa: too few buffer samples (%d), skipping.", len(all_exps))
        return -1.0

    device = accelerator.device
    texts = [
        build_judge_prompt(
            [], e.question, e.strategy, context_text_raw=e.context_text
        )
        for e in all_exps
    ]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)

    judge_model.eval()
    with torch.no_grad():
        logits = accelerator.unwrap_model(judge_model)(**enc)   # [N]
        scores = torch.sigmoid(logits).cpu().tolist()
    judge_model.train()

    high_score = [(s, e.outcome) for s, e in zip(scores, all_exps) if s > 0.6]
    if not high_score:
        return 0.0
    joa = sum(1 for _, y in high_score if y == 1) / len(high_score)
    logger.info("JOA: %.3f  (%d / %d high-score samples)", joa, sum(1 for _, y in high_score if y == 1), len(high_score))
    return joa


# ---------------------------------------------------------------------------
# Weight sync: Actor FSDP → vLLM reload
# ---------------------------------------------------------------------------

def save_actor_for_vllm(actor, accelerator: Accelerator, out_dir: str) -> None:
    """Save the FSDP-wrapped Actor as HuggingFace safetensors for vLLM reload."""
    if accelerator.is_main_process:
        os.makedirs(out_dir, exist_ok=True)
        accelerator.unwrap_model(actor).save_pretrained(
            out_dir, safe_serialization=True
        )
        logger.info("Saved Actor weights to %s for vLLM reload.", out_dir)
    accelerator.wait_for_everyone()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: ActorJudgeConfig) -> None:
    cfg.validate()

    # ── Accelerator (FSDP) ───────────────────────────────────────────────────
    accelerator = Accelerator()
    device = accelerator.device

    # ── WandB (main process only) ────────────────────────────────────────────
    if accelerator.is_main_process:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=cfg.__dict__)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.start_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # M6: Register <|judge|> special token and warm-start its embedding
    tokenizer.add_tokens(["<|judge|>"])

    # ── Models ───────────────────────────────────────────────────────────────
    logger.info("Loading Actor from %s", cfg.start_model_path)
    actor = AutoModelForCausalLM.from_pretrained(
        cfg.start_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    logger.info("Loading Reference Model (frozen)")
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.start_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    ref_model.requires_grad_(False)

    logger.info("Loading Judge from Actor checkpoint")
    judge = JudgeModel.from_actor_checkpoint(cfg.start_model_path, torch_dtype=torch.bfloat16)

    # M6: Resize Judge embedding table and warm-start <|judge|>
    judge.transformer.resize_token_embeddings(len(tokenizer))
    warmstart_judge_token_embedding(judge, tokenizer, source_token="<|im_end|>")

    # ── Accelerate FSDP distribution ─────────────────────────────────────────
    # P5: ALL three models must be prepared to avoid per-rank full copies (64GB waste).
    actor, ref_model, judge = (
        accelerator.prepare(actor),
        accelerator.prepare(ref_model),
        accelerator.prepare(judge),
    )

    # ── Trainers & buffer ────────────────────────────────────────────────────
    buffer = UCBBuffer(
        max_size=cfg.buffer_max_size,
        per_q_max=cfg.per_q_max,
        lambda_err=cfg.lambda_err,
        lambda_exp=cfg.lambda_exp,
        disable_ucb=cfg.disable_ucb_replay,
    )

    actor_trainer = ActorTrainer(actor, ref_model, tokenizer, accelerator, cfg)
    judge_trainer = JudgeTrainer(judge, tokenizer, accelerator, cfg)
    rollout_engine = RolloutEngine(tokenizer, cfg)

    # ── DataLoader ───────────────────────────────────────────────────────────
    train_dataset = load_rollout_dataset(cfg)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        collate_fn=lambda x: x,   # keep as list of RolloutSample
    )

    # ── Step 0: Judge Warmup (optional) ──────────────────────────────────────
    if cfg.judge_warmup and accelerator.is_main_process:
        logger.info("=== Judge Warmup (%d steps) ===", cfg.warmup_steps)
        warmup_samples = [s for s in train_dataset if s.s_gold is not None][:cfg.warmup_steps * 4]
        if warmup_samples:
            for step in range(cfg.warmup_steps):
                batch_w = random.sample(warmup_samples, min(cfg.judge_batch_size, len(warmup_samples)))
                s_golds   = [s.s_gold for s in batch_w]
                questions = [s.question for s in batch_w]
                # Simple negatives: shuffle chars of gold strategy
                s_negs    = [s[::-1] for s in s_golds]
                judge_trainer.warmup_step(s_golds, s_negs, questions, global_step=step)
        else:
            logger.warning("No S_gold data found — skipping Judge warmup.")

    accelerator.wait_for_everyone()

    # ── Ray init ─────────────────────────────────────────────────────────────
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    current_model_path = cfg.start_model_path
    global_step = 0
    avg_strategy_lengths: List[float] = []   # for Length Hacking monitoring

    # ── Phase II Main Loop ───────────────────────────────────────────────────
    for epoch in range(cfg.total_epochs):
        logger.info("=== Epoch %d / %d ===", epoch + 1, cfg.total_epochs)

        for batch_idx, batch in enumerate(train_loader):
            global_step += 1
            experiences = None

            # ── M1: Only Rank 0 launches vLLM ────────────────────────────
            if accelerator.is_main_process:
                vllm_actor = VLLMActor.remote(
                    current_model_path,
                    tensor_parallel_size=cfg.tensor_parallel_size,
                    gpu_memory_utilization=cfg.gpu_memory_utilization,
                )
                experiences = rollout_engine.run(batch, vllm_actor, global_step)
                # Kill vLLM immediately to release GPU memory before FSDP training
                ray.kill(vllm_actor)
                del vllm_actor

            # ── Sync all ranks; wait for vLLM GPU memory to be freed ──────
            accelerator.wait_for_everyone()
            torch.cuda.empty_cache()

            # ── Broadcast experiences from Rank 0 to all ranks ───────────
            # P4: experiences contain only pure-Python scalars — pickle-safe
            experiences = broadcast_object_list_from_rank0(experiences, accelerator)
            buffer.extend(experiences)

            # Track average strategy length for Length Hacking detection
            lengths = [len(e.strategy.split()) for e in experiences if e.outcome >= 0]
            if lengths:
                avg_strategy_lengths.append(sum(lengths) / len(lengths))

            # ── Step 2: Judge update (OFF-POLICY, uses buffer history) ────
            judge_loss = 0.0
            if not cfg.freeze_judge and len(buffer) >= cfg.min_buffer_size:
                judge_loss = judge_trainer.train_step(buffer, global_step)

            # ── Step 3+4: Actor GRPO update (ON-POLICY, uses experiences) ─
            actor_loss = actor_trainer.train_step(experiences, judge, global_step)

            # ── Logging ───────────────────────────────────────────────────
            if accelerator.is_main_process:
                n_correct = sum(1 for e in experiences if e.outcome == 1)
                n_valid   = sum(1 for e in experiences if e.outcome >= 0)
                avg_len   = avg_strategy_lengths[-1] if avg_strategy_lengths else 0.0
                metrics = {
                    "train/actor_loss":        actor_loss,
                    "train/judge_loss":        judge_loss,
                    "train/pass_rate":         n_correct / max(n_valid, 1),
                    "train/buffer_size":       len(buffer),
                    "train/avg_strategy_len":  avg_len,
                    "train/global_step":       global_step,
                }
                wandb.log(metrics, step=global_step)
                logger.info(
                    "step=%d actor_loss=%.4f judge_loss=%.4f pass=%.2f buf=%d avg_len=%.0f",
                    global_step, actor_loss, judge_loss,
                    metrics["train/pass_rate"], len(buffer), avg_len,
                )

                # Length Hacking early warning
                win = cfg.length_hack_window
                if len(avg_strategy_lengths) >= win + 1:
                    recent    = avg_strategy_lengths[-win:]
                    reference = avg_strategy_lengths[-(win + 1)]
                    if all(recent[i] > recent[i-1] * 1.10 for i in range(1, win)):
                        logger.warning(
                            "⚠ Length Hacking alert: avg_strategy_length rose >10%% "
                            "for %d consecutive steps (%.0f → %.0f). "
                            "Consider enabling length_penalty_coeff.",
                            win, reference, recent[-1],
                        )

        # ── End of epoch ──────────────────────────────────────────────────

        # Save Actor weights → write to /dev/shm for vLLM reload next epoch
        save_actor_for_vllm(actor, accelerator, cfg.weight_sync_tmp_dir)
        current_model_path = cfg.weight_sync_tmp_dir

        # Checkpoint
        if epoch % cfg.save_freq == 0 and accelerator.is_main_process:
            ckpt_dir = os.path.join(cfg.checkpoint_dir, f"epoch_{epoch:03d}")
            accelerator.unwrap_model(actor).save_pretrained(ckpt_dir)
            logger.info("Saved Actor checkpoint to %s", ckpt_dir)

        # Validation: Pass@1 + JOA
        if epoch % cfg.val_freq == 0:
            for subdir in cfg.val_subdirs:
                val_ds = load_val_dataset(cfg, subdir)
                pass1  = validate_pass_at_1(
                    actor, tokenizer, val_ds, cfg, accelerator, subdir
                )
                if accelerator.is_main_process:
                    wandb.log({f"val/{subdir}/pass@1": pass1}, step=global_step)

            # JOA (plan §5): Judge-Outcome Agreement on buffer samples
            # Measures whether the Judge correctly assigns high scores to y=1
            if not cfg.freeze_judge and len(buffer) >= cfg.min_buffer_size:
                joa = compute_joa(judge, tokenizer, buffer, cfg, accelerator)
                if accelerator.is_main_process and joa >= 0:
                    wandb.log({"val/joa": joa}, step=global_step)
                    logger.info("epoch %d JOA=%.3f", epoch, joa)

    if accelerator.is_main_process:
        wandb.finish()
    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Actor-Judge Phase II Training")
    # Allow overriding any config field via --key value
    parser.add_argument("--actor_sft_checkpoint", type=str, default="")
    parser.add_argument("--actor_model_path",     type=str,
                        default="/home/test/test16/chenlu/model/Qwen3-4B")
    parser.add_argument("--total_epochs",  type=int,   default=5)
    parser.add_argument("--K",             type=int,   default=8)
    parser.add_argument("--alpha",         type=float, default=0.3)
    parser.add_argument("--freeze_judge",  action="store_true")
    parser.add_argument("--dense_reward_alpha", type=float, default=0.3)
    parser.add_argument("--disable_ucb_replay", action="store_true")
    parser.add_argument("--wandb_run_name", type=str, default="phase2_co_evolution")
    args = parser.parse_args()

    cfg = ActorJudgeConfig(
        actor_sft_checkpoint=args.actor_sft_checkpoint,
        actor_model_path=args.actor_model_path,
        total_epochs=args.total_epochs,
        K=args.K,
        alpha=args.alpha,
        dense_reward_alpha=args.dense_reward_alpha,
        freeze_judge=args.freeze_judge,
        disable_ucb_replay=args.disable_ucb_replay,
        wandb_run_name=args.wandb_run_name,
    )

    main(cfg)
