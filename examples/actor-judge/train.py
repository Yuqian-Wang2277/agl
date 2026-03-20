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

import dataclasses
import datetime
import json
import logging
import os
import random
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from env import evaluate
from prompts import (
    apply_chat_template,
    build_answer_prompt,
    build_judge_prompt,
    build_strategy_prompt,
    STRATEGY_CLOSE,
    STRATEGY_OPEN,
)
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
# Config snapshot + checkpoint pruning
# ---------------------------------------------------------------------------


def save_config_snapshot(cfg: ActorJudgeConfig, out_path: Path) -> None:
    """Persist full ``ActorJudgeConfig`` for reproducibility (no WandB-only reliance)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dataclasses.asdict(cfg)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def prune_epoch_checkpoints(cfg: ActorJudgeConfig, metrics_records: List[Dict[str, Any]]) -> None:
    """Remove ``epoch_*`` dirs not in keep_last_k ∪ keep_best_k (by mean strict Pass@1)."""
    if cfg.keep_last_k_checkpoints <= 0 and cfg.keep_best_k_checkpoints <= 0:
        return
    root = Path(cfg.checkpoint_dir)
    if not root.is_dir():
        return
    epoch_dirs: List[Tuple[int, Path]] = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("epoch_"):
            try:
                ep = int(p.name.split("_", 1)[1])
                epoch_dirs.append((ep, p))
            except ValueError:
                continue
    if not epoch_dirs:
        return
    by_ep = {ep: p for ep, p in epoch_dirs}
    epochs_sorted = sorted(by_ep.keys())
    keep: set[int] = set()
    if cfg.keep_last_k_checkpoints > 0:
        keep.update(epochs_sorted[-cfg.keep_last_k_checkpoints :])
    if cfg.keep_best_k_checkpoints > 0 and metrics_records:
        ranked = sorted(
            (m for m in metrics_records if int(m.get("epoch", -1)) >= 0),
            key=lambda m: float(m.get("mean_pass_at_1", -1.0)),
            reverse=True,
        )
        for m in ranked[: cfg.keep_best_k_checkpoints]:
            keep.add(int(m["epoch"]))
    for ep, p in by_ep.items():
        if ep not in keep:
            logger.info("Pruning old checkpoint directory: %s", p)
            shutil.rmtree(p, ignore_errors=True)


# ---------------------------------------------------------------------------
# Validation — vLLM greedy Pass@1 (strict denominator; matches val token caps)
# ---------------------------------------------------------------------------


def _attach_judge_scores_to_val_items(
    items: List[Dict[str, Any]],
    judge: torch.nn.Module,
    tokenizer,
    accelerator: Accelerator,
    batch_size: int,
    freeze_judge: bool,
) -> None:
    """In-place: add judge_logit / judge_prob per item (rank-0 FSDP unwrap)."""
    if not items:
        return
    if freeze_judge:
        for it in items:
            it["judge_logit"] = None
            it["judge_prob"] = None
        return

    judge_u = accelerator.unwrap_model(judge)
    was_training = judge_u.training
    judge_u.eval()
    device = accelerator.device

    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        texts = [
            build_judge_prompt([], it["question"], it["strategy_text"], context_text_raw="")
            for it in chunk
        ]
        enc = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        )
        enc = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            logits = judge_u(**enc).float().reshape(-1)
        probs = torch.sigmoid(logits).cpu().tolist()
        log_list = logits.cpu().tolist()
        for j, it in enumerate(chunk):
            it["judge_logit"] = float(log_list[j])
            it["judge_prob"] = float(probs[j])

    if was_training:
        judge_u.train()


def _validate_split_pass1_vllm(
    tokenizer,
    cfg: ActorJudgeConfig,
    vllm_actor,
    val_dataset: ActorJudgeDataset,
    split_name: str,
) -> Tuple[float, int, int, List[Dict[str, Any]]]:
    """Strict Pass@1: denominator = every problem; format errors (-1) count as incorrect.

    Returns (pass1, correct, n, items) where *items* is empty unless cfg.val_save_item_details.
    """
    from vllm import SamplingParams

    samples = list(val_dataset)
    n = len(samples)
    if n == 0:
        return 0.0, 0, 0, []

    prompts1 = [
        apply_chat_template(tokenizer, build_strategy_prompt(s.fewshot_examples))
        for s in samples
    ]
    sp1 = SamplingParams(
        temperature=0.0,
        max_tokens=cfg.val_strategy_max_tokens,
        stop=[STRATEGY_CLOSE],
    )
    raw1 = ray.get(vllm_actor.generate.remote(prompts1, sp1))
    s_texts = [t + STRATEGY_CLOSE for t in raw1]

    prompts2: List[str] = []
    idx2: List[int] = []
    for i, (s_text, sample) in enumerate(zip(s_texts, samples)):
        if STRATEGY_OPEN in s_text and STRATEGY_CLOSE in s_text:
            prompts2.append(
                apply_chat_template(tokenizer, build_answer_prompt(s_text, sample.question))
            )
            idx2.append(i)

    sp2 = SamplingParams(
        temperature=0.0,
        max_tokens=cfg.val_answer_max_tokens,
        stop=["</answer>"],
    )
    answers: List[str] = [""] * n
    idx_to_stage2_prompt: Dict[int, str] = {}
    if prompts2:
        raw2 = ray.get(vllm_actor.generate.remote(prompts2, sp2))
        for j, i in enumerate(idx2):
            answers[i] = raw2[j] + "</answer>"
            idx_to_stage2_prompt[i] = prompts2[j]

    items: List[Dict[str, Any]] = []
    correct = 0
    for i, sample in enumerate(samples):
        outcome = evaluate(s_texts[i], answers[i], sample.answer_gold)
        if outcome == 1:
            correct += 1
        if cfg.val_save_item_details:
            p2 = idx_to_stage2_prompt.get(i, "")
            items.append(
                {
                    "split": split_name,
                    "index_in_split": i,
                    "domain": sample.domain,
                    "question": sample.question,
                    "answer_gold": sample.answer_gold,
                    "stage1_prompt": prompts1[i],
                    "stage2_prompt": p2,
                    "strategy_text": s_texts[i],
                    "answer_text": answers[i],
                    "outcome": int(outcome),
                }
            )

    pass1 = correct / max(n, 1)
    logger.info(
        "Validation [%s] strict Pass@1 = %.4f (%d/%d)",
        split_name,
        pass1,
        correct,
        n,
    )
    return pass1, correct, n, items


def run_full_validation_vllm(
    tokenizer,
    cfg: ActorJudgeConfig,
    model_path: str,
    judge: torch.nn.Module,
    buffer: UCBBuffer,
    accelerator: Accelerator,
    epoch: int,
    global_step: int,
) -> Optional[Dict[str, Any]]:
    """vLLM Pass@1 on rank-0; JOA on all ranks (FSDP). Returns metrics dict on rank-0 only."""
    split_results: Dict[str, Any] = {}
    pass_vals: List[float] = []
    all_items: List[Dict[str, Any]] = []
    mean_p = 0.0

    if accelerator.is_main_process:
        logger.info(
            "vLLM validation: TP=%d strat_tok=%d ans_tok=%d samples/split=%d",
            cfg.tensor_parallel_size,
            cfg.val_strategy_max_tokens,
            cfg.val_answer_max_tokens,
            cfg.val_num_samples,
        )
        vllm_actor = VLLMActor.remote(
            model_path,
            tensor_parallel_size=cfg.tensor_parallel_size,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
        )
        try:
            for subdir in cfg.val_subdirs:
                val_ds = load_val_dataset(cfg, subdir)
                p1, c, n, items = _validate_split_pass1_vllm(
                    tokenizer, cfg, vllm_actor, val_ds, subdir
                )
                row: Dict[str, Any] = {"pass_at_1": p1, "correct": c, "total": n}
                if items:
                    row["items"] = items
                split_results[subdir] = row
                pass_vals.append(p1)
                all_items.extend(items)
        finally:
            ray.kill(vllm_actor)

        mean_p = sum(pass_vals) / max(len(pass_vals), 1)
        logger.info("Validation finished: mean strict Pass@1 = %.4f", mean_p)

    if accelerator.num_processes > 1:
        bundle_in = None
        if accelerator.is_main_process:
            bundle_in = {
                "split_results": split_results,
                "mean_pass_at_1": mean_p,
                "all_items": all_items,
            }
        b = broadcast_object_list_from_rank0(bundle_in, accelerator)
        assert b is not None
        split_results = b["split_results"]
        mean_p = b["mean_pass_at_1"]
        all_items = b["all_items"]

    if cfg.val_save_item_details and all_items:
        _attach_judge_scores_to_val_items(
            all_items,
            judge,
            tokenizer,
            accelerator,
            cfg.val_judge_score_batch_size,
            cfg.freeze_judge,
        )

    accelerator.wait_for_everyone()

    joa_val: Optional[float] = None
    if not cfg.freeze_judge and len(buffer) >= cfg.min_buffer_size:
        joa_val = compute_joa(judge, tokenizer, buffer, cfg, accelerator)

    if not accelerator.is_main_process:
        return None

    pack: Dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "splits": split_results,
        "mean_pass_at_1": mean_p,
        "joa": joa_val,
        "val_strategy_max_tokens": cfg.val_strategy_max_tokens,
        "val_answer_max_tokens": cfg.val_answer_max_tokens,
        "denominator": "all_items_strict",
        "val_save_item_details": cfg.val_save_item_details,
    }
    return pack


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
    actor,
    judge,
    actor_trainer,
    judge_trainer,
    epoch: int,
    global_step: int,
    cfg: ActorJudgeConfig,
    accelerator: Accelerator,
    tokenizer,
    eval_results: Optional[Dict[str, Any]] = None,
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
        torch.save(
            actor_trainer.optimizer.state_dict(),
            os.path.join(ckpt_dir, "actor_optimizer.pt"),
        )
        torch.save(
            judge_trainer.optimizer.state_dict(),
            os.path.join(ckpt_dir, "judge_optimizer.pt"),
        )
        # Training metadata
        with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
            json.dump({"epoch": epoch, "global_step": global_step}, f)
        if eval_results is not None:
            with open(os.path.join(ckpt_dir, "eval_results.json"), "w") as f:
                json.dump(eval_results, f, indent=2, default=str)
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
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
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
    # Permanently freeze ref_model in eval mode so Dropout layers (if any in
    # Qwen3 attention) never fire during the KL forward pass.  Without eval(),
    # requires_grad_(False) alone prevents weight updates but does NOT disable
    # stochastic behaviour — the KL baseline would jitter every step and cause
    # the Actor to diverge.
    ref_model.eval()

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

    if accelerator.is_main_process:
        save_config_snapshot(cfg, Path(cfg.checkpoint_dir) / "config.json")
        logger.info("Wrote config snapshot to %s", Path(cfg.checkpoint_dir) / "config.json")

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

                    # Build few-shot context for each sample (aligns warmup
                    # distribution with Phase II where Judge always receives
                    # context_text_raw from rollout_engine).
                    warmup_contexts = []
                    for s in batch_w:
                        lines = []
                        for ex in (s.fewshot_examples or []):
                            inp = ex.get("input", "")
                            tgt = ex.get("target", "")
                            if isinstance(tgt, list):
                                tgt = tgt[0] if tgt else ""
                            lines.append(f"Q: {inp}  A: {tgt}")
                        warmup_contexts.append("\n".join(lines))

                    judge_trainer.warmup_step(
                        s_golds,
                        s_negs,
                        questions,
                        global_step=step,
                        advance_scheduler=False,
                        context_texts=warmup_contexts,
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
    val_metrics_history: List[Dict[str, Any]] = []

    # Sync Actor to HF dir for vLLM (rollout + validation share this path)
    save_actor_for_vllm(actor, tokenizer, accelerator, cfg.weight_sync_tmp_dir)
    current_model_path = cfg.weight_sync_tmp_dir
    accelerator.wait_for_everyone()

    # Baseline strict Pass@1 (+ fail-fast) before any RL epoch
    if cfg.val_before_train and start_epoch == 0 and cfg.val_freq > 0:
        base_pack = run_full_validation_vllm(
            tokenizer,
            cfg,
            cfg.weight_sync_tmp_dir,
            judge,
            buffer,
            accelerator,
            epoch=-1,
            global_step=0,
        )
        if accelerator.is_main_process and base_pack is not None:
            with open(Path(cfg.checkpoint_dir) / "eval_baseline.json", "w") as f:
                json.dump(base_pack, f, indent=2, default=str)
            for sub, row in base_pack["splits"].items():
                wandb.log(
                    {f"val_baseline/{sub}/pass@1_strict": row["pass_at_1"]},
                    step=0,
                )
            wandb.log(
                {"val_baseline/mean_pass@1_strict": base_pack["mean_pass_at_1"]},
                step=0,
            )
            j_b = base_pack.get("joa")
            if j_b is not None and j_b >= 0:
                wandb.log({"val_baseline/joa": j_b}, step=0)
            val_metrics_history.append(
                {"epoch": -1, "mean_pass_at_1": base_pack["mean_pass_at_1"]}
            )
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()

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

        # ── Save Actor weights to /dev/shm (vLLM reload + validation) ───────
        save_actor_for_vllm(actor, tokenizer, accelerator, cfg.weight_sync_tmp_dir)
        current_model_path = cfg.weight_sync_tmp_dir

        eval_pack: Optional[Dict[str, Any]] = None
        if epoch % cfg.val_freq == 0:
            eval_pack = run_full_validation_vllm(
                tokenizer,
                cfg,
                cfg.weight_sync_tmp_dir,
                judge,
                buffer,
                accelerator,
                epoch=epoch,
                global_step=global_step,
            )
            if accelerator.is_main_process and eval_pack is not None:
                ep_json = Path(cfg.checkpoint_dir) / f"eval_epoch_{epoch:03d}.json"
                with open(ep_json, "w") as f:
                    json.dump(eval_pack, f, indent=2, default=str)
                for sub, row in eval_pack["splits"].items():
                    wandb.log(
                        {f"val/{sub}/pass@1_strict": row["pass_at_1"]},
                        step=global_step,
                    )
                wandb.log(
                    {"val/mean_pass@1_strict": eval_pack["mean_pass_at_1"]},
                    step=global_step,
                )
                j_e = eval_pack.get("joa")
                if j_e is not None and j_e >= 0:
                    wandb.log({"val/joa": j_e}, step=global_step)
                    logger.info("epoch %d JOA=%.3f", epoch, j_e)
                val_metrics_history.append(
                    {
                        "epoch": epoch,
                        "mean_pass_at_1": eval_pack["mean_pass_at_1"],
                    }
                )
            accelerator.wait_for_everyone()
            torch.cuda.empty_cache()

        if epoch % cfg.save_freq == 0:
            save_checkpoint(
                actor,
                judge,
                actor_trainer,
                judge_trainer,
                epoch,
                global_step,
                cfg,
                accelerator,
                tokenizer,
                eval_results=eval_pack,
            )

        if (
            epoch % cfg.val_freq == 0
            and (cfg.keep_last_k_checkpoints > 0 or cfg.keep_best_k_checkpoints > 0)
            and accelerator.is_main_process
        ):
            prune_epoch_checkpoints(cfg, val_metrics_history)

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
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default="./checkpoints_actor_judge",
        help="Parent directory; each run writes to checkpoint_root/run_name/",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Subfolder for this run (weights + config.json). Empty → timestamp.",
    )
    parser.add_argument(
        "--no_val_before_train",
        action="store_true",
        help="Skip baseline vLLM validation before epoch 0",
    )
    parser.add_argument("--val_strategy_max_tokens", type=int, default=2048)
    parser.add_argument("--val_answer_max_tokens", type=int, default=512)
    parser.add_argument("--val_num_samples", type=int, default=500)
    parser.add_argument("--keep_last_k_checkpoints", type=int, default=2)
    parser.add_argument("--keep_best_k_checkpoints", type=int, default=2)
    parser.add_argument(
        "--no_val_save_item_details",
        action="store_true",
        help="Omit per-item prompts/generations from eval_*.json (smaller files)",
    )
    parser.add_argument("--val_judge_score_batch_size", type=int, default=16)
    args = parser.parse_args()

    run_name = args.run_name.strip() or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(args.checkpoint_root, run_name)

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
        checkpoint_root=args.checkpoint_root,
        run_name=run_name,
        checkpoint_dir=checkpoint_dir,
        val_before_train=not args.no_val_before_train,
        val_strategy_max_tokens=args.val_strategy_max_tokens,
        val_answer_max_tokens=args.val_answer_max_tokens,
        val_num_samples=args.val_num_samples,
        keep_last_k_checkpoints=args.keep_last_k_checkpoints,
        keep_best_k_checkpoints=args.keep_best_k_checkpoints,
        val_save_item_details=not args.no_val_save_item_details,
        val_judge_score_batch_size=args.val_judge_score_batch_size,
    )

    main(cfg)
