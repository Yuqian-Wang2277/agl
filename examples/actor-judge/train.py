"""Phase II main training loop — Actor-Judge Co-Evolution.

Launch with:
    accelerate launch --config_file accelerate_fsdp.yaml train.py

Architecture and all review-round fixes incorporated:

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

S1  Epoch-level vLLM lifecycle:
    vLLM is created ONCE per epoch for Phase A (all rollout batches), then
    killed ONCE before Phase B (all training batches).  The previous per-batch
    kill/restart would cost ~15 s × 2500 batches ≈ 10+ hours/epoch in startup.

C3  Judge warmup runs on ALL ranks (not just rank 0).
    Previously guarded by is_main_process, causing a deadlock: rank 0 called
    accel.backward() while ranks 1-7 waited at wait_for_everyone().

C4  validate_pass_at_1 loads the saved HF checkpoint on rank 0 only.
    FSDP ZeRO-3 shards parameters across ranks; calling generate() on the
    unwrapped (sharded) model produces garbage.  Loading the saved full-param
    checkpoint avoids all FSDP complexity.

C5  broadcast_object_list is guarded by num_processes > 1.
    torch.distributed is not initialised in single-GPU runs (debug mode).

S5  Both Actor tokenizer and Judge model weights are saved each checkpoint.
    The tokenizer has <|judge|> added; saving it ensures consistent vocab on
    reload.  The Judge is never needed for deployment but is needed for resume.

S6  Judge warmup uses cross-domain gold strategies as negatives.
    String-reversal ("txet ygetarts") is trivially distinguishable and teaches
    the Judge nothing about strategy quality.  Cross-domain strategies are
    grammatically valid but semantically wrong for the question.

M4  avg_strategy_lengths uses a bounded deque (no unbounded memory growth).
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from prompts import build_judge_prompt
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

    C5 fix: Only called when num_processes > 1.  torch.distributed is not
    initialised in single-GPU debug runs, so calling this unconditionally
    would crash with "Default process group has not been initialized".

    P4: obj must contain NO GPU Tensors — only pure Python scalars.
    """
    container = [obj]
    torch.distributed.broadcast_object_list(container, src=0)
    return container[0]


# ---------------------------------------------------------------------------
# Judge warmup: mode (cold / always / reuse), eval, checkpoint I/O
# ---------------------------------------------------------------------------


def effective_judge_warmup_mode(cfg: ActorJudgeConfig) -> str:
    """``judge_warmup=False`` keeps backward-compat by forcing cold start."""
    if not cfg.judge_warmup:
        return "cold"
    return (cfg.judge_warmup_mode or "always").strip().lower()


def resolve_judge_checkpoint_file(path: str) -> Path:
    """Return path to ``judge_model.pt`` (file or inside a directory)."""
    p = Path(path).expanduser()
    if p.is_file() and p.suffix == ".pt":
        return p
    if p.is_dir():
        cand = p / "judge_model.pt"
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"Judge checkpoint not found: expected a .pt file or a directory "
        f"containing judge_model.pt, got {path!r}"
    )


def load_judge_weights_only(
    judge: torch.nn.Module, accelerator: Accelerator, ckpt_path: str
) -> None:
    """Load Judge parameters only (no optimizer). For reuse / resume distinction."""
    path = resolve_judge_checkpoint_file(ckpt_path)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    unwrapped = accelerator.unwrap_model(judge)
    missing, unexpected = unwrapped.load_state_dict(state, strict=False)
    logger.info(
        "Loaded Judge weights from %s (strict=False; missing=%d unexpected=%d)",
        path,
        len(missing),
        len(unexpected),
    )


def judge_warmup_save_dir(cfg: ActorJudgeConfig) -> Path:
    if (cfg.judge_warmup_save_dir or "").strip():
        return Path(cfg.judge_warmup_save_dir).expanduser()
    return Path(cfg.checkpoint_dir) / "judge_warmup_latest"


def save_judge_warmup_weights(
    judge: torch.nn.Module,
    accelerator: Accelerator,
    out_dir: Path,
    meta: Optional[dict] = None,
) -> None:
    out_dir = Path(out_dir)
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            accelerator.unwrap_model(judge).state_dict(),
            out_dir / "judge_model.pt",
        )
        payload = {"kind": "judge_warmup", "weights_only": True}
        if meta:
            payload.update(meta)
        with open(out_dir / "warmup_meta.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved Judge warmup weights to %s", out_dir / "judge_model.pt")
    accelerator.wait_for_everyone()


def sample_warmup_negative(
    s,
    s_gold: str,
    version: str,
    domain_to_gold_by_version: Dict[str, Dict[str, str]],
    rng: random.Random,
) -> Tuple[str, str, str]:
    """Return (negative_strategy, negative_domain_label, negative_kind)."""
    domain_to_gold = domain_to_gold_by_version.get(version, {})
    others = [d for d in domain_to_gold if d != s.domain]
    if others:
        neg_domain = rng.choice(others)
        return domain_to_gold[neg_domain], neg_domain, "cross_domain"
    words = s_gold.split()
    w = words[:]
    rng.shuffle(w)
    return " ".join(w), "word_shuffle", "word_shuffle"


def _warmup_batch_to_device(batch, device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


def judge_warmup_pairwise_metrics(
    judge: torch.nn.Module,
    tokenizer,
    device: torch.device,
    triples: List[Tuple[str, str, str]],
    batch_size: int = 16,
) -> Tuple[float, float]:
    """Return (acc, avg_margin): acc uses logit_win > logit_lose; margin = mean(sigmoid(w)-sigmoid(l))."""
    if not triples:
        return 0.0, 0.0
    judge.eval()
    n_ok = 0
    n_tot = 0
    margin_sum = 0.0

    with torch.no_grad():
        for i in range(0, len(triples), batch_size):
            chunk = triples[i : i + batch_size]
            qs = [t[0] for t in chunk]
            pos = [t[1] for t in chunk]
            neg = [t[2] for t in chunk]
            win_texts = [build_judge_prompt([], q, sp) for q, sp in zip(qs, pos)]
            lose_texts = [build_judge_prompt([], q, sn) for q, sn in zip(qs, neg)]
            inputs_win = _warmup_batch_to_device(
                tokenizer(
                    win_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                ),
                device,
            )
            inputs_lose = _warmup_batch_to_device(
                tokenizer(
                    lose_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                ),
                device,
            )
            lw = judge(**inputs_win).float()
            ll = judge(**inputs_lose).float()
            n_ok += int((lw > ll).sum().item())
            n_tot += len(chunk)
            margin_sum += (torch.sigmoid(lw) - torch.sigmoid(ll)).sum().item()
    judge.train()
    acc = n_ok / max(n_tot, 1)
    avg_margin = margin_sum / max(n_tot, 1)
    return acc, avg_margin


# ---------------------------------------------------------------------------
# Validation (C4 fix: load saved HF model on rank-0, not FSDP-wrapped actor)
# ---------------------------------------------------------------------------

def validate_pass_at_1(
    model_path: str,               # C4 fix: path to saved HF checkpoint
    tokenizer,
    val_dataset: ActorJudgeDataset,
    cfg: ActorJudgeConfig,
    accelerator: Accelerator,
    split_name: str,
) -> Optional[float]:
    """Greedy-decode Pass@1 on *val_dataset*, executed on Rank 0 only.

    C4 fix: loads the saved HF checkpoint from *model_path* for inference.
    This avoids FSDP ZeRO-3 issues where the unwrapped model has only the
    local parameter shard and generate() produces garbage.

    The saved model already exists (written by save_actor_for_vllm which runs
    at end of epoch, before this function is called).

    Returns None on non-rank-0 processes.
    """
    from prompts import apply_chat_template, build_strategy_prompt, build_answer_prompt
    from env import evaluate

    if not accelerator.is_main_process:
        return None

    # Load saved full-parameter model (not FSDP-sharded) on rank 0
    val_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(accelerator.device)
    val_model.eval()

    correct = 0
    total   = 0

    # M2: batch validation for speed (batch_size=4 to save memory)
    val_batch_size = 4
    samples = list(val_dataset)

    for i in range(0, len(samples), val_batch_size):
        batch = samples[i : i + val_batch_size]

        # Stage-1: strategy generation (batch)
        prompts1 = [
            apply_chat_template(tokenizer, build_strategy_prompt(s.fewshot_examples))
            for s in batch
        ]
        enc1 = tokenizer(prompts1, return_tensors="pt", padding=True,
                         truncation=True, max_length=2048).to(accelerator.device)
        with torch.no_grad():
            out1 = val_model.generate(
                **enc1,
                max_new_tokens=cfg.strategy_max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        s_texts = []
        for j in range(len(batch)):
            generated = tokenizer.decode(
                out1[j][enc1["input_ids"].shape[1]:], skip_special_tokens=False
            )
            s_texts.append(generated + "</strategy>")

        # Stage-2: answer generation (batch)
        prompts2 = [
            apply_chat_template(tokenizer, build_answer_prompt(s_text, sample.question))
            for s_text, sample in zip(s_texts, batch)
        ]
        enc2 = tokenizer(prompts2, return_tensors="pt", padding=True,
                         truncation=True, max_length=2048).to(accelerator.device)
        with torch.no_grad():
            out2 = val_model.generate(
                **enc2,
                max_new_tokens=cfg.answer_max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, sample in enumerate(batch):
            a_text = tokenizer.decode(
                out2[j][enc2["input_ids"].shape[1]:], skip_special_tokens=False
            ) + "</answer>"
            outcome = evaluate(s_texts[j], a_text, sample.answer_gold)
            if outcome == 1:
                correct += 1
            if outcome >= 0:
                total += 1

    del val_model
    torch.cuda.empty_cache()

    pass1 = correct / max(total, 1)
    logger.info("Validation [%s]: Pass@1 = %.3f (%d/%d)", split_name, pass1, correct, total)
    return pass1


# ---------------------------------------------------------------------------
# JOA: Judge-Outcome Agreement (plan §5)
# ---------------------------------------------------------------------------

def compute_joa(
    judge_model: torch.nn.Module,
    tokenizer,
    buffer: UCBBuffer,
    cfg: ActorJudgeConfig,
    accelerator: Accelerator,
    n_samples: int = 200,
) -> float:
    """JOA = fraction(y=1 | σ(v_judge) > 0.6) measured on buffer samples.

    Uses the FSDP-wrapped judge_model directly (not unwrap_model) so that
    FSDP all-gather is triggered correctly.  All ranks participate.

    Returns JOA value in [0, 1], or -1.0 if buffer data is insufficient.
    """
    from prompts import build_judge_prompt

    all_exps = buffer.sample_individual(n_samples)
    if len(all_exps) < 10:
        logger.warning("compute_joa: too few buffer samples (%d), skipping.", len(all_exps))
        return -1.0

    device = accelerator.device
    texts = [
        build_judge_prompt([], e.question, e.strategy, context_text_raw=e.context_text)
        for e in all_exps
    ]
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=2048
    ).to(device)

    judge_model.eval()
    with torch.no_grad():
        logits = judge_model(**enc)   # use FSDP-wrapped directly (C2-style fix)
        scores = torch.sigmoid(logits).cpu().tolist()
    judge_model.train()

    high_score = [(s, e.outcome) for s, e in zip(scores, all_exps) if s > 0.6]
    if not high_score:
        return 0.0
    n_correct = sum(1 for _, y in high_score if y == 1)
    joa = n_correct / len(high_score)
    logger.info("JOA: %.3f  (%d / %d high-score samples)", joa, n_correct, len(high_score))
    return joa


# ---------------------------------------------------------------------------
# Weight sync: Actor FSDP → vLLM reload
# ---------------------------------------------------------------------------

def save_actor_for_vllm(
    actor: torch.nn.Module,
    tokenizer,
    accelerator: Accelerator,
    out_dir: str,
) -> None:
    """Save the FSDP-wrapped Actor as HuggingFace safetensors for vLLM reload.

    S5 fix: also saves the tokenizer (which has <|judge|> added).
    """
    if accelerator.is_main_process:
        os.makedirs(out_dir, exist_ok=True)
        accelerator.unwrap_model(actor).save_pretrained(
            out_dir, safe_serialization=True
        )
        tokenizer.save_pretrained(out_dir)   # S5 fix: save tokenizer with new vocab
        logger.info("Saved Actor + tokenizer to %s for vLLM reload.", out_dir)
    accelerator.wait_for_everyone()


# ---------------------------------------------------------------------------
# Checkpoint save / load (L5: basic resume support)
# ---------------------------------------------------------------------------

def save_checkpoint(
    actor, judge, actor_trainer, judge_trainer,
    epoch: int, global_step: int,
    cfg: ActorJudgeConfig, accelerator: Accelerator,
    tokenizer,
) -> None:
    """Save full training state for potential resume."""
    ckpt_dir = os.path.join(cfg.checkpoint_dir, f"epoch_{epoch:03d}")
    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)
        # Actor weights
        accelerator.unwrap_model(actor).save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        # S5 fix: Judge weights
        torch.save(
            accelerator.unwrap_model(judge).state_dict(),
            os.path.join(ckpt_dir, "judge_model.pt"),
        )
        # Optimizer + scheduler states (for resume)
        torch.save(actor_trainer.optimizer.state_dict(),
                   os.path.join(ckpt_dir, "actor_optimizer.pt"))
        torch.save(judge_trainer.optimizer.state_dict(),
                   os.path.join(ckpt_dir, "judge_optimizer.pt"))
        # Training metadata
        with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
            json.dump({"epoch": epoch, "global_step": global_step}, f)
        logger.info("Saved full checkpoint to %s", ckpt_dir)
    accelerator.wait_for_everyone()


def try_resume(
    actor, judge, actor_trainer, judge_trainer,
    cfg: ActorJudgeConfig, accelerator: Accelerator,
) -> tuple:
    """Attempt to resume from cfg.resume_from_checkpoint. Returns (start_epoch, global_step)."""
    if not cfg.resume_from_checkpoint or not os.path.isdir(cfg.resume_from_checkpoint):
        return 0, 0

    ckpt_dir = cfg.resume_from_checkpoint
    state_file = os.path.join(ckpt_dir, "training_state.json")
    if not os.path.exists(state_file):
        logger.warning("Resume requested but training_state.json not found in %s", ckpt_dir)
        return 0, 0

    with open(state_file) as f:
        state = json.load(f)
    start_epoch  = state["epoch"] + 1
    global_step  = state["global_step"]

    actor_opt_path = os.path.join(ckpt_dir, "actor_optimizer.pt")
    judge_opt_path = os.path.join(ckpt_dir, "judge_optimizer.pt")
    if os.path.exists(actor_opt_path):
        actor_trainer.optimizer.load_state_dict(torch.load(actor_opt_path, map_location="cpu"))
    if os.path.exists(judge_opt_path):
        judge_trainer.optimizer.load_state_dict(torch.load(judge_opt_path, map_location="cpu"))

    logger.info("Resumed from %s: start_epoch=%d global_step=%d", ckpt_dir, start_epoch, global_step)
    return start_epoch, global_step


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

    # M6: Register <|judge|> special token (must be done before model loads)
    tokenizer.add_tokens(["<|judge|>"])

    # ── Models ───────────────────────────────────────────────────────────────
    logger.info("Loading Actor from %s", cfg.start_model_path)
    actor = AutoModelForCausalLM.from_pretrained(
        cfg.start_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    # Resize actor embedding table to match new tokenizer (good practice even
    # though <|judge|> only appears in Judge inputs, not Actor inputs)
    actor.resize_token_embeddings(len(tokenizer))

    logger.info("Loading Reference Model (frozen)")
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.start_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    ref_model.resize_token_embeddings(len(tokenizer))
    ref_model.requires_grad_(False)

    logger.info("Loading Judge from Actor checkpoint")
    judge = JudgeModel.from_actor_checkpoint(cfg.start_model_path, torch_dtype=torch.bfloat16)

    # M6: Resize Judge embedding table and warm-start <|judge|>
    judge.transformer.resize_token_embeddings(len(tokenizer))
    warmstart_judge_token_embedding(judge, tokenizer, source_token="<|im_end|>")

    # ── Accelerate FSDP distribution ─────────────────────────────────────────
    # P5: ALL three models must be prepared to avoid per-rank full copies (64 GB waste).
    actor, ref_model, judge = (
        accelerator.prepare(actor),
        accelerator.prepare(ref_model),
        accelerator.prepare(judge),
    )

    # ── Dataset & DataLoader ──────────────────────────────────────────────────
    train_dataset = load_rollout_dataset(cfg)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        collate_fn=lambda x: x,   # keep as list of RolloutSample
    )

    # L4: auto-compute total_train_steps for LR schedulers if not set
    steps_per_epoch = len(train_loader)
    if cfg.total_train_steps == 0:
        cfg.total_train_steps = cfg.total_epochs * steps_per_epoch
        logger.info("auto total_train_steps = %d", cfg.total_train_steps)

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

    # ── L5: Resume from checkpoint if requested ───────────────────────────────
    start_epoch, global_step = try_resume(
        actor, judge, actor_trainer, judge_trainer, cfg, accelerator
    )

    # ── Step 0: Judge Warmup (cold / always / reuse) ───────────────────────────
    # C3 fix: ALL ranks participate — no is_main_process guard.
    # ``resume_from_checkpoint`` restores optimizers only; reuse loads Judge weights
    # from ``judge_init_checkpoint`` with a fresh Phase II optimizer.
    warm_mode = effective_judge_warmup_mode(cfg)
    logger.info("Judge warmup mode = %s", warm_mode)
    if cfg.resume_from_checkpoint and warm_mode == "reuse" and cfg.judge_init_checkpoint:
        logger.warning(
            "resume_from_checkpoint is set together with judge_warmup_mode=reuse. "
            "Optimizer state is restored from resume, then Judge weights are "
            "overwritten from judge_init_checkpoint — verify this is intended."
        )

    if warm_mode == "reuse":
        load_judge_weights_only(judge, accelerator, cfg.judge_init_checkpoint.strip())
        judge_trainer.rebuild_optimizer_for_phase2()
        logger.info("Judge reuse: loaded weights, Phase II optimizer reset.")

    elif warm_mode == "cold":
        logger.info("Judge cold start: no warmup weights loaded.")

    elif warm_mode == "always":
        warmup_lr = cfg.judge_warmup_lr if cfg.judge_warmup_lr > 0 else cfg.judge_lr * 0.1

        rng = random.Random(cfg.warmup_seed)
        warmup_samples = [
            s
            for s in train_dataset
            if (s.s_gold_by_version and len(s.s_gold_by_version) > 0) or s.s_gold is not None
        ]

        warmup_dump_path = Path(cfg.checkpoint_dir) / "judge_warmup_pairs.jsonl"
        if accelerator.is_main_process:
            warmup_dump_path.parent.mkdir(parents=True, exist_ok=True)
            if warmup_dump_path.exists():
                warmup_dump_path.unlink()
            logger.info("Judge warmup pairs will be saved to %s", warmup_dump_path)

        if not warmup_samples:
            logger.warning("No S_gold data found — skipping Judge warmup.")
        else:
            judge_trainer.set_optimizer_lr(warmup_lr)
            logger.info(
                "=== Judge Warmup (always): up to %d steps, lr=%.2e "
                "(Phase II judge_lr=%.2e) ===",
                cfg.warmup_steps,
                warmup_lr,
                cfg.judge_lr,
            )
            version_to_pool_full = {"v1": [], "v2": [], "v3": []}
            for s in warmup_samples:
                if s.s_gold_by_version:
                    for v, txt in s.s_gold_by_version.items():
                        if txt:
                            version_to_pool_full[v].append((s, txt))
                elif s.s_gold:
                    version_to_pool_full["v1"].append((s, s.s_gold))

            flat: List[Tuple[object, str, str]] = []
            for v in ("v1", "v2", "v3"):
                for pair in version_to_pool_full[v]:
                    flat.append((pair[0], pair[1], v))

            rng.shuffle(flat)
            if len(flat) < 32:
                eval_flat = []
                train_flat = flat[:]
            else:
                n_eval = int(len(flat) * cfg.judge_warmup_eval_ratio)
                n_eval = max(16, min(n_eval, len(flat) // 4, 512))
                if len(flat) - n_eval < cfg.judge_batch_size:
                    n_eval = min(max(0, len(flat) // 10), max(0, len(flat) - 1))
                eval_flat = flat[:n_eval] if n_eval > 0 else []
                train_flat = flat[n_eval:] if n_eval > 0 else flat[:]

            while (
                eval_flat
                and len(train_flat) < cfg.judge_batch_size
                and len(train_flat) < len(flat)
            ):
                train_flat.insert(0, eval_flat.pop())

            version_to_pool = {"v1": [], "v2": [], "v3": []}
            for s, txt, v in train_flat:
                version_to_pool[v].append((s, txt))

            active_versions = [v for v, pool in version_to_pool.items() if pool]
            logger.info(
                "Judge warmup: train_pairs=%d eval_pairs=%d version_counts=%s",
                len(train_flat),
                len(eval_flat),
                {vx: len(version_to_pool[vx]) for vx in ("v1", "v2", "v3")},
            )

            domain_to_gold_by_version: Dict[str, Dict[str, str]] = {"v1": {}, "v2": {}, "v3": {}}
            for s in warmup_samples:
                if s.s_gold_by_version:
                    for v, txt in s.s_gold_by_version.items():
                        if txt and s.domain not in domain_to_gold_by_version[v]:
                            domain_to_gold_by_version[v][s.domain] = txt
                elif s.s_gold and s.domain not in domain_to_gold_by_version["v1"]:
                    domain_to_gold_by_version["v1"][s.domain] = s.s_gold

            eval_triples: List[Tuple[str, str, str]] = []
            for s, pos, v in eval_flat:
                rng_e = random.Random((cfg.warmup_seed ^ hash(s.question) ^ hash(v)) % (2**31))
                neg, _, _ = sample_warmup_negative(
                    s, pos, v, domain_to_gold_by_version, rng_e
                )
                eval_triples.append((s.question, pos, neg))

            early_stopped = False
            last_step = -1
            if not train_flat:
                logger.warning(
                    "Warmup: no training pairs after eval split; skipping warmup updates."
                )
            else:
                for step in range(cfg.warmup_steps):
                    last_step = step
                    if active_versions:
                        target_version = active_versions[step % len(active_versions)]
                    else:
                        target_version = "v1"

                    pool = version_to_pool.get(target_version, [])
                    if not pool:
                        logger.warning(
                            "Warmup step %d: no samples for %s, skipping.",
                            step,
                            target_version,
                        )
                        continue

                    batch_pairs = rng.sample(pool, min(cfg.judge_batch_size, len(pool)))
                    batch_w = [p[0] for p in batch_pairs]
                    s_golds = [p[1] for p in batch_pairs]
                    questions = [s.question for s in batch_w]

                    s_negs: List[str] = []
                    neg_domains: List[str] = []
                    for s, s_gold in zip(batch_w, s_golds):
                        neg, neg_domain, _ = sample_warmup_negative(
                            s, s_gold, target_version, domain_to_gold_by_version, rng
                        )
                        s_negs.append(neg)
                        neg_domains.append(neg_domain)

                    if accelerator.is_main_process:
                        with open(warmup_dump_path, "a", encoding="utf-8") as f:
                            for s, q, pos, neg, neg_domain in zip(
                                batch_w, questions, s_golds, s_negs, neg_domains
                            ):
                                f.write(
                                    json.dumps(
                                        {
                                            "step": step,
                                            "version": target_version,
                                            "domain": s.domain,
                                            "negative_domain": neg_domain,
                                            "question": q,
                                            "positive_strategy": pos,
                                            "negative_strategy": neg,
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )

                    judge_trainer.warmup_step(
                        s_golds,
                        s_negs,
                        questions,
                        global_step=step,
                        advance_scheduler=False,
                    )

                    if (
                        eval_triples
                        and cfg.judge_warmup_eval_every > 0
                        and (step + 1) % cfg.judge_warmup_eval_every == 0
                    ):
                        acc, avg_margin = judge_warmup_pairwise_metrics(
                            judge,
                            tokenizer,
                            accelerator.device,
                            eval_triples,
                            batch_size=min(16, cfg.judge_batch_size),
                        )
                        if accelerator.is_main_process:
                            wandb.log(
                                {
                                    "warmup/eval_pairwise_acc": acc,
                                    "warmup/eval_avg_margin": avg_margin,
                                },
                                step=step,
                            )
                        logger.info(
                            "Warmup eval step %d: pairwise_acc=%.4f avg_margin=%.4f (n_eval=%d)",
                            step,
                            acc,
                            avg_margin,
                            len(eval_triples),
                        )
                        if acc >= cfg.judge_warmup_overfit_warn_acc:
                            logger.warning(
                                "Warmup eval acc %.3f >= overfit_warn %.3f — "
                                "Judge may be overfitting trivial cues; monitor Phase II.",
                                acc,
                                cfg.judge_warmup_overfit_warn_acc,
                            )
                        if (
                            acc >= cfg.judge_warmup_early_stop_min_acc
                            and avg_margin >= cfg.judge_warmup_early_stop_min_margin
                        ):
                            logger.info(
                                "Warmup early stop: acc %.3f >= %.3f and avg_margin %.3f >= %.3f",
                                acc,
                                cfg.judge_warmup_early_stop_min_acc,
                                avg_margin,
                                cfg.judge_warmup_early_stop_min_margin,
                            )
                            early_stopped = True
                            break

            out_dir = judge_warmup_save_dir(cfg)
            save_judge_warmup_weights(
                judge,
                accelerator,
                out_dir,
                meta={
                    "warmup_steps_done": 0 if last_step < 0 else last_step + 1,
                    "early_stopped": early_stopped,
                    "eval_pairs": len(eval_triples),
                    "warmup_seed": cfg.warmup_seed,
                },
            )

            if cfg.judge_warmup_reset_optimizer_after:
                judge_trainer.rebuild_optimizer_for_phase2()
                logger.info("Judge warmup done: Phase II optimizer re-initialised.")
            else:
                judge_trainer.set_optimizer_lr(cfg.judge_lr)
                logger.info(
                    "Judge warmup done: kept optimizer state; lr reset to %.2e",
                    cfg.judge_lr,
                )

    accelerator.wait_for_everyone()

    # ── Ray init ─────────────────────────────────────────────────────────────
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    current_model_path = cfg.start_model_path
    # M4 fix: bounded deque instead of unbounded list (avoids memory growth over
    # many epochs × many steps × float per entry)
    avg_strategy_lengths: deque = deque(maxlen=cfg.length_hack_window * 10)

    # ── Phase II Main Loop ───────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.total_epochs):
        logger.info("=== Epoch %d / %d ===", epoch + 1, cfg.total_epochs)

        # ════════════════════════════════════════════════════════════════════
        # Phase A: Full-epoch Rollout  (S1 fix: vLLM created/killed ONCE per epoch)
        #
        # Previously vLLM was created and killed on EVERY batch:
        #   for batch: create vLLM (15s) → rollout → kill → train → repeat
        # With 2500 batches/epoch: 2500 × 15s = 10+ hours/epoch just in startup.
        #
        # Now we collect all rollout experiences first, then train:
        #   Phase A: create vLLM → all batches → kill vLLM  (one 15s startup)
        #   Phase B: all training steps
        # ════════════════════════════════════════════════════════════════════
        epoch_experiences: List = []

        if accelerator.is_main_process:
            logger.info("Phase A: starting vLLM rollout ...")
            vllm_actor = VLLMActor.remote(
                current_model_path,
                tensor_parallel_size=cfg.tensor_parallel_size,
                gpu_memory_utilization=cfg.gpu_memory_utilization,
            )
            for rollout_step, batch in enumerate(train_loader):
                exps = rollout_engine.run(batch, vllm_actor, epoch * steps_per_epoch + rollout_step)
                epoch_experiences.extend(exps)

            logger.info(
                "Phase A done: %d experiences collected. Killing vLLM ...",
                len(epoch_experiences),
            )
            ray.kill(vllm_actor)
            del vllm_actor

        # All ranks sync; wait for vLLM GPU memory to be released
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()

        # ── Broadcast all epoch experiences to non-rank-0 processes ──────────
        # C5 fix: guard with num_processes > 1 (single-GPU debug has no dist)
        if accelerator.num_processes > 1:
            epoch_experiences = broadcast_object_list_from_rank0(epoch_experiences, accelerator)
        # else: single-GPU, epoch_experiences already populated on rank 0

        buffer.extend(epoch_experiences)

        # Track strategy lengths for Length Hacking detection
        epoch_lens = [len(e.strategy.split()) for e in epoch_experiences if e.outcome >= 0]
        if epoch_lens:
            avg_len_epoch = sum(epoch_lens) / len(epoch_lens)
            avg_strategy_lengths.append(avg_len_epoch)

        # ════════════════════════════════════════════════════════════════════
        # Phase B: Training   (no vLLM — all GPU memory freed)
        # Chunk epoch_experiences into B×K batches for Judge + Actor updates.
        # ════════════════════════════════════════════════════════════════════
        BK = cfg.train_batch_size * cfg.K
        experience_batches = [
            epoch_experiences[i : i + BK]
            for i in range(0, len(epoch_experiences), BK)
        ]

        logger.info("Phase B: training on %d batches ...", len(experience_batches))

        epoch_actor_loss = 0.0
        epoch_judge_loss = 0.0

        for batch_exps in experience_batches:
            global_step += 1

            # Step 2: Judge update (OFF-POLICY, uses buffer history)
            judge_loss = 0.0
            if not cfg.freeze_judge and len(buffer) >= cfg.min_buffer_size:
                judge_loss = judge_trainer.train_step(buffer, global_step)

            # Step 3+4: Actor GRPO update (ON-POLICY, uses current rollout batch)
            actor_loss = actor_trainer.train_step(batch_exps, judge, global_step)

            epoch_actor_loss += actor_loss
            epoch_judge_loss += judge_loss

            # ── Per-step logging ──────────────────────────────────────────
            if accelerator.is_main_process and global_step % 50 == 0:
                n_correct = sum(1 for e in batch_exps if e.outcome == 1)
                n_valid   = sum(1 for e in batch_exps if e.outcome >= 0)
                metrics = {
                    "train/actor_loss":        actor_loss,
                    "train/judge_loss":        judge_loss,
                    "train/pass_rate":         n_correct / max(n_valid, 1),
                    "train/buffer_size":       len(buffer),
                    "train/global_step":       global_step,
                }
                wandb.log(metrics, step=global_step)

        # ── End-of-epoch logging ──────────────────────────────────────────
        if accelerator.is_main_process and experience_batches:
            n_batches = len(experience_batches)
            avg_len = avg_strategy_lengths[-1] if avg_strategy_lengths else 0.0
            wandb.log({
                "epoch/actor_loss":        epoch_actor_loss / n_batches,
                "epoch/judge_loss":        epoch_judge_loss / n_batches,
                "epoch/avg_strategy_len":  avg_len,
                "epoch/buffer_size":       len(buffer),
            }, step=global_step)

            # Length Hacking early warning
            win = cfg.length_hack_window
            lengths_list = list(avg_strategy_lengths)
            if len(lengths_list) >= win + 1:
                recent    = lengths_list[-win:]
                reference = lengths_list[-(win + 1)]
                if all(recent[i] > recent[i - 1] * 1.10 for i in range(1, win)):
                    logger.warning(
                        "⚠ Length Hacking alert: avg_strategy_length rose >10%% "
                        "for %d consecutive epochs (%.0f → %.0f). "
                        "Consider enabling length_penalty_coeff.",
                        win, reference, recent[-1],
                    )

        # ── Save Actor weights to /dev/shm for vLLM reload next epoch ────────
        save_actor_for_vllm(actor, tokenizer, accelerator, cfg.weight_sync_tmp_dir)
        current_model_path = cfg.weight_sync_tmp_dir

        # ── Full checkpoint (actor + judge + optimizers) ──────────────────────
        if epoch % cfg.save_freq == 0:
            save_checkpoint(
                actor, judge, actor_trainer, judge_trainer,
                epoch, global_step, cfg, accelerator, tokenizer,
            )

        # ── Validation: Pass@1 + JOA ──────────────────────────────────────────
        if epoch % cfg.val_freq == 0:
            for subdir in cfg.val_subdirs:
                val_ds  = load_val_dataset(cfg, subdir)
                # C4 fix: validate using saved HF model on rank 0 only
                pass1   = validate_pass_at_1(
                    cfg.weight_sync_tmp_dir,   # path to saved HF checkpoint
                    tokenizer, val_ds, cfg, accelerator, subdir
                )
                if accelerator.is_main_process and pass1 is not None:
                    wandb.log({f"val/{subdir}/pass@1": pass1}, step=global_step)

            # JOA (plan §5): Judge-Outcome Agreement on buffer samples
            if not cfg.freeze_judge and len(buffer) >= cfg.min_buffer_size:
                joa = compute_joa(judge, tokenizer, buffer, cfg, accelerator)
                if accelerator.is_main_process and joa >= 0:
                    wandb.log({"val/joa": joa}, step=global_step)
                    logger.info("epoch %d JOA=%.3f", epoch, joa)

        # All ranks must sync before next epoch (save / val may be rank-0 only)
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        wandb.finish()
    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Actor-Judge Phase II Training")
    parser.add_argument("--actor_sft_checkpoint", type=str, default="")
    parser.add_argument("--actor_model_path",     type=str,
                        default="/home/test/test16/chenlu/model/Qwen3-4B")
    parser.add_argument("--total_epochs",          type=int,   default=5)
    parser.add_argument("--num_train_samples",     type=int,   default=20_000)
    parser.add_argument("--K",                     type=int,   default=8)
    parser.add_argument("--alpha",                 type=float, default=0.3)
    parser.add_argument("--freeze_judge",          action="store_true")
    parser.add_argument("--dense_reward_alpha",    type=float, default=0.3)
    parser.add_argument("--disable_ucb_replay",    action="store_true")
    parser.add_argument("--wandb_run_name",        type=str, default="phase2_co_evolution")
    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    parser.add_argument(
        "--judge_warmup_mode",
        type=str,
        default="always",
        choices=["cold", "always", "reuse"],
        help="cold=no warmup; always=run warmup (+eval/early-stop) then Phase II; "
        "reuse=load judge_init_checkpoint weights only, fresh optimizer",
    )
    parser.add_argument(
        "--judge_init_checkpoint",
        type=str,
        default="",
        help="judge_model.pt or directory containing it (required for mode=reuse)",
    )
    parser.add_argument("--judge_warmup_save_dir", type=str, default="")
    parser.add_argument("--warmup_seed", type=int, default=42)
    parser.add_argument("--judge_warmup_lr", type=float, default=0.0)
    parser.add_argument("--judge_warmup_eval_ratio", type=float, default=0.12)
    parser.add_argument("--judge_warmup_eval_every", type=int, default=5)
    parser.add_argument("--judge_warmup_early_stop_min_acc", type=float, default=0.75)
    parser.add_argument("--judge_warmup_early_stop_min_margin", type=float, default=0.15)
    parser.add_argument("--judge_warmup_overfit_warn_acc", type=float, default=0.95)
    parser.add_argument(
        "--no_judge_warmup_reset_optimizer",
        action="store_true",
        help="Keep Adam state after warmup (default: reset optimizer for Phase II)",
    )
    parser.add_argument(
        "--no_judge_warmup",
        action="store_true",
        help="Legacy: set judge_warmup=False (forces cold start regardless of mode)",
    )
    args = parser.parse_args()

    cfg = ActorJudgeConfig(
        actor_sft_checkpoint=args.actor_sft_checkpoint,
        actor_model_path=args.actor_model_path,
        total_epochs=args.total_epochs,
        num_train_samples=args.num_train_samples,
        K=args.K,
        alpha=args.alpha,
        dense_reward_alpha=args.dense_reward_alpha,
        freeze_judge=args.freeze_judge,
        disable_ucb_replay=args.disable_ucb_replay,
        wandb_run_name=args.wandb_run_name,
        resume_from_checkpoint=args.resume_from_checkpoint,
        judge_warmup_mode=args.judge_warmup_mode,
        judge_init_checkpoint=args.judge_init_checkpoint,
        judge_warmup_save_dir=args.judge_warmup_save_dir,
        warmup_seed=args.warmup_seed,
        judge_warmup_lr=args.judge_warmup_lr,
        judge_warmup_eval_ratio=args.judge_warmup_eval_ratio,
        judge_warmup_eval_every=args.judge_warmup_eval_every,
        judge_warmup_early_stop_min_acc=args.judge_warmup_early_stop_min_acc,
        judge_warmup_early_stop_min_margin=args.judge_warmup_early_stop_min_margin,
        judge_warmup_overfit_warn_acc=args.judge_warmup_overfit_warn_acc,
        judge_warmup_reset_optimizer_after=not args.no_judge_warmup_reset_optimizer,
        judge_warmup=not args.no_judge_warmup,
    )

    main(cfg)
