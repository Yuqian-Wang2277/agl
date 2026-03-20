"""Standalone Judge warmup runner (pairwise reward-model supervision).

This script pretrains Judge only, then saves:
1) a reusable Judge weights file: judge_model.pt
2) a HuggingFace backbone checkpoint (for inspection / reproducibility)
3) tokenizer files
4) warmup metadata + sampled train pairs
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ActorJudgeConfig
from data_loader import load_rollout_dataset
from judge_model import JudgeModel, warmstart_judge_token_embedding
from judge_encode import encode_batch_for_judge
from prompts import build_judge_prompt_body


def sample_negative(
    sample,
    s_gold: str,
    version: str,
    domain_to_gold_by_version: Dict[str, Dict[str, str]],
    rng: random.Random,
) -> Tuple[str, str, str]:
    domain_to_gold = domain_to_gold_by_version.get(version, {})
    others = [d for d in domain_to_gold if d != sample.domain]
    if others:
        d = rng.choice(others)
        return domain_to_gold[d], d, "cross_domain"
    words = s_gold.split()
    rng.shuffle(words)
    return " ".join(words), "word_shuffle", "word_shuffle"


def _batch_to_device(batch, device: torch.device) -> Dict[str, torch.Tensor]:
    """Move tokenizer outputs to ``device`` (BatchEncoding.to can be unreliable)."""
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


def eval_pairwise_metrics(
    judge: JudgeModel,
    tokenizer,
    device: torch.device,
    triples: List[Tuple[str, str, str]],
    batch_size: int = 16,
    scores_path: Optional[Path] = None,
    step: int = -1,
    judge_max_length: int = 8192,
) -> Tuple[float, float]:
    """Pairwise eval: accuracy via raw logits (monotonic with sigmoid); margin = mean(sigmoid(w)-sigmoid(l)).

    Returns:
        (acc, avg_margin) with acc = fraction where logit_win > logit_lose.
    """
    if not triples:
        return 0.0, 0.0
    judge.eval()
    ok = 0
    total = 0
    margin_sum = 0.0
    pair_base = 0
    with torch.no_grad():
        for i in range(0, len(triples), batch_size):
            chunk = triples[i : i + batch_size]
            qs = [x[0] for x in chunk]
            pos = [x[1] for x in chunk]
            neg = [x[2] for x in chunk]
            win_b = [build_judge_prompt_body([], q, s) for q, s in zip(qs, pos)]
            lose_b = [build_judge_prompt_body([], q, s) for q, s in zip(qs, neg)]
            in_win = _batch_to_device(
                encode_batch_for_judge(tokenizer, win_b, judge_max_length),
                device,
            )
            in_lose = _batch_to_device(
                encode_batch_for_judge(tokenizer, lose_b, judge_max_length),
                device,
            )
            lw = judge(**in_win).float()
            ll = judge(**in_lose).float()
            ok += int((lw > ll).sum().item())
            total += len(chunk)
            pw = torch.sigmoid(lw)
            pl = torch.sigmoid(ll)
            margins = pw - pl
            margin_sum += margins.sum().item()
            if scores_path is not None:
                margins_list = margins.detach().cpu().tolist()
                correct_list = (lw > ll).detach().cpu().tolist()
                lw_list = lw.detach().cpu().tolist()
                ll_list = ll.detach().cpu().tolist()
                pw_list = pw.detach().cpu().tolist()
                pl_list = pl.detach().cpu().tolist()
                with open(scores_path, "a", encoding="utf-8") as sf:
                    for j, (q, sp, sn) in enumerate(zip(qs, pos, neg)):
                        idx = pair_base + j
                        sf.write(
                            json.dumps(
                                {
                                    "step": step,
                                    "pair_idx": idx,
                                    "logit_win": lw_list[j],
                                    "logit_lose": ll_list[j],
                                    "prob_win": pw_list[j],
                                    "prob_lose": pl_list[j],
                                    "margin": margins_list[j],
                                    "correct": bool(correct_list[j]),
                                    "question": q,
                                    "positive_strategy": sp,
                                    "negative_strategy": sn,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
            pair_base += len(chunk)
    judge.train()
    acc = ok / max(total, 1)
    avg_margin = margin_sum / max(total, 1)
    return acc, avg_margin


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Judge warmup")
    parser.add_argument("--actor_sft_checkpoint", type=str, default="")
    parser.add_argument("--actor_model_path", type=str, default="/home/test/test16/chenlu/model/Qwen3-4B")
    parser.add_argument("--output_dir", type=str, default="./judge_warmup_ckpt")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--judge_batch_size", type=int, default=16)
    parser.add_argument("--judge_warmup_lr", type=float, default=1e-6)
    parser.add_argument("--judge_warmup_eval_ratio", type=float, default=0.12)
    parser.add_argument("--judge_warmup_eval_every", type=int, default=5)
    parser.add_argument("--judge_warmup_early_stop_min_acc", type=float, default=0.75)
    parser.add_argument(
        "--judge_warmup_early_stop_min_margin",
        type=float,
        default=0.15,
        help="Mean prob(win)-prob(lose) on held-out pairs; early stop requires acc and margin.",
    )
    parser.add_argument("--judge_warmup_overfit_warn_acc", type=float, default=0.95)
    parser.add_argument("--warmup_seed", type=int, default=42)
    parser.add_argument("--num_train_samples", type=int, default=20000)
    parser.add_argument("--train_subdir", type=str, default="train_20k")
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Transformer device_map for Judge backbone (e.g. auto)",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help="Use low_cpu_mem_usage=True when loading HF weights.",
    )
    args = parser.parse_args()

    cfg = ActorJudgeConfig(
        actor_sft_checkpoint=args.actor_sft_checkpoint,
        actor_model_path=args.actor_model_path,
        warmup_steps=args.warmup_steps,
        judge_batch_size=args.judge_batch_size,
        judge_warmup_lr=args.judge_warmup_lr,
        judge_warmup_eval_ratio=args.judge_warmup_eval_ratio,
        judge_warmup_eval_every=args.judge_warmup_eval_every,
        judge_warmup_early_stop_min_acc=args.judge_warmup_early_stop_min_acc,
        judge_warmup_early_stop_min_margin=args.judge_warmup_early_stop_min_margin,
        judge_warmup_overfit_warn_acc=args.judge_warmup_overfit_warn_acc,
        warmup_seed=args.warmup_seed,
        num_train_samples=args.num_train_samples,
        train_subdir=args.train_subdir,
    )
    cfg.validate()

    rng = random.Random(cfg.warmup_seed)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Explicit device / device_map logging ──────────────────────────
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"[warmup] CUDA_VISIBLE_DEVICES={cuda_visible!r}")
    # NOTE:
    # We intentionally avoid calling `torch.cuda.is_available()` /
    # `torch.cuda.device_count()` here. In this environment those calls can
    # intermittently trigger `cudaGetDeviceCount` initialization errors and
    # cause subsequent model loading to fall back to CPU.
    # We will instead rely on HF `hf_device_map` and module device/dtype
    # logging right after `from_pretrained`.

    tokenizer = AutoTokenizer.from_pretrained(cfg.start_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(["<|judge|>"])

    # Load backbone with HF sharding so warmup can use multiple GPUs.
    causal_lm = AutoModelForCausalLM.from_pretrained(
        cfg.start_model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )

    print(f"[warmup] loaded backbone with device_map={args.device_map!r}")
    hf_device_map = getattr(causal_lm, "hf_device_map", None)
    if hf_device_map:
        items = list(hf_device_map.items())[:40]
        print(f"[warmup] hf_device_map (first {len(items)}/{len(hf_device_map)} items):")
        for k, v in items:
            print(f"  - {k}: {v}")
    else:
        print("[warmup] hf_device_map not found on model instance")

    if hasattr(causal_lm, "model"):
        backbone = causal_lm.model
    else:
        backbone = causal_lm.transformer

    hidden_size = causal_lm.config.hidden_size
    del causal_lm  # keep only the backbone reference

    judge = JudgeModel(backbone, hidden_size)
    judge.scalar_head.to(dtype=torch.bfloat16)

    judge.transformer.resize_token_embeddings(len(tokenizer))
    warmstart_judge_token_embedding(judge, tokenizer, source_token="<|im_end|>")
    judge.train()

    # Move inputs to the embedding device (first executed device in hf device_map).
    model_device = judge.transformer.get_input_embeddings().weight.device

    head_w = judge.scalar_head.weight
    print(f"[warmup] judge.embedding: device={model_device}, dtype={judge.transformer.get_input_embeddings().weight.dtype}")
    print(f"[warmup] judge.scalar_head: device={head_w.device}, dtype={head_w.dtype}")

    optimizer = torch.optim.AdamW(
        [p for p in judge.parameters() if p.requires_grad],
        lr=cfg.judge_warmup_lr if cfg.judge_warmup_lr > 0 else cfg.judge_lr * 0.1,
    )

    ds = load_rollout_dataset(
        cfg,
        tokenizer=tokenizer,
        stage1_reject_log_path=str(out_dir / "dataset_stage1_rejects.jsonl"),
    )
    warmup_samples = [
        s for s in ds if (s.s_gold_by_version and len(s.s_gold_by_version) > 0) or s.s_gold is not None
    ]
    if not warmup_samples:
        raise RuntimeError("No warmup samples with s_gold found.")

    version_to_pool = {"v1": [], "v2": [], "v3": []}
    for s in warmup_samples:
        if s.s_gold_by_version:
            for v, txt in s.s_gold_by_version.items():
                if txt:
                    version_to_pool[v].append((s, txt))
        elif s.s_gold:
            version_to_pool["v1"].append((s, s.s_gold))
    active_versions = [v for v, pool in version_to_pool.items() if pool]
    if not active_versions:
        raise RuntimeError("No usable warmup pools for v1/v2/v3.")

    domain_to_gold_by_version: Dict[str, Dict[str, str]] = {"v1": {}, "v2": {}, "v3": {}}
    for s in warmup_samples:
        if s.s_gold_by_version:
            for v, txt in s.s_gold_by_version.items():
                if txt and s.domain not in domain_to_gold_by_version[v]:
                    domain_to_gold_by_version[v][s.domain] = txt
        elif s.s_gold and s.domain not in domain_to_gold_by_version["v1"]:
            domain_to_gold_by_version["v1"][s.domain] = s.s_gold

    flat = []
    for v in ("v1", "v2", "v3"):
        for s, txt in version_to_pool[v]:
            flat.append((s, txt, v))
    rng.shuffle(flat)
    if len(flat) < 32:
        eval_flat, train_flat = [], flat[:]
    else:
        n_eval = int(len(flat) * cfg.judge_warmup_eval_ratio)
        n_eval = max(16, min(n_eval, len(flat) // 4, 512))
        eval_flat, train_flat = flat[:n_eval], flat[n_eval:]

    eval_triples = []
    for s, pos, v in eval_flat:
        r = random.Random((cfg.warmup_seed ^ hash(s.question) ^ hash(v)) % (2**31))
        neg, _, _ = sample_negative(s, pos, v, domain_to_gold_by_version, r)
        eval_triples.append((s.question, pos, neg))

    pairs_path = out_dir / "judge_warmup_pairs.jsonl"
    if pairs_path.exists():
        pairs_path.unlink()
    scores_path = out_dir / "judge_warmup_eval_scores.jsonl"
    if scores_path.exists():
        scores_path.unlink()

    best_eval = -1.0
    best_eval_margin = -1.0
    best_step = -1
    early_stopped = False
    max_grad_norm = cfg.max_grad_norm

    for step in range(cfg.warmup_steps):
        version = active_versions[step % len(active_versions)]
        pool = version_to_pool[version]
        batch_pairs = rng.sample(pool, min(cfg.judge_batch_size, len(pool)))
        batch_s = [p[0] for p in batch_pairs]
        s_golds = [p[1] for p in batch_pairs]
        questions = [s.question for s in batch_s]
        s_negs, neg_domains = [], []
        for s, pos in zip(batch_s, s_golds):
            neg, neg_d, _ = sample_negative(s, pos, version, domain_to_gold_by_version, rng)
            s_negs.append(neg)
            neg_domains.append(neg_d)

        with open(pairs_path, "a", encoding="utf-8") as f:
            for s, q, pos, neg, neg_d in zip(batch_s, questions, s_golds, s_negs, neg_domains):
                f.write(json.dumps({
                    "step": step,
                    "version": version,
                    "domain": s.domain,
                    "negative_domain": neg_d,
                    "question": q,
                    "positive_strategy": pos,
                    "negative_strategy": neg,
                }, ensure_ascii=False) + "\n")

        jmax = int(getattr(cfg, "judge_max_length", 8192))
        win_bodies = [build_judge_prompt_body([], q, s) for q, s in zip(questions, s_golds)]
        lose_bodies = [build_judge_prompt_body([], q, s) for q, s in zip(questions, s_negs)]
        inputs_win = _batch_to_device(
            encode_batch_for_judge(tokenizer, win_bodies, jmax),
            model_device,
        )
        inputs_lose = _batch_to_device(
            encode_batch_for_judge(tokenizer, lose_bodies, jmax),
            model_device,
        )

        if step == 0:
            print("[warmup] === step 0 runtime probes (before forward) ===")
            print(f"[warmup] inputs_win.input_ids.device={inputs_win['input_ids'].device}")
            print(f"[warmup] inputs_win.input_ids.dtype={inputs_win['input_ids'].dtype}")
            print(f"[warmup] inputs_win.attention_mask.device={inputs_win['attention_mask'].device}")
            print(f"[warmup] inputs_win.attention_mask.dtype={inputs_win['attention_mask'].dtype}")

        optimizer.zero_grad()
        logit_win = judge(**inputs_win)
        logit_lose = judge(**inputs_lose)

        if step == 0:
            print("[warmup] === step 0 runtime probes (after forward) ===")
            print(f"[warmup] scalar_head: device={judge.scalar_head.weight.device}, dtype={judge.scalar_head.weight.dtype}")
            print(f"[warmup] logit_win: device={logit_win.device}, dtype={logit_win.dtype}")
            print(f"[warmup] logit_lose: device={logit_lose.device}, dtype={logit_lose.dtype}")
        bt_loss = -torch.nn.functional.logsigmoid(logit_win - logit_lose).mean()
        l2_penalty = 0.001 * (logit_win**2 + logit_lose**2).mean()
        loss = bt_loss + l2_penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(judge.parameters(), max_grad_norm)
        optimizer.step()

        if eval_triples and cfg.judge_warmup_eval_every > 0 and (step + 1) % cfg.judge_warmup_eval_every == 0:
            acc, avg_margin = eval_pairwise_metrics(
                judge,
                tokenizer,
                model_device,
                eval_triples,
                batch_size=min(16, cfg.judge_batch_size),
                scores_path=scores_path,
                step=step,
                judge_max_length=cfg.judge_max_length,
            )
            print(
                f"[warmup] step={step} loss={loss.item():.4f} "
                f"eval_acc={acc:.4f} eval_avg_margin={avg_margin:.4f}"
            )
            if acc > best_eval or (acc == best_eval and avg_margin > best_eval_margin):
                best_eval = acc
                best_eval_margin = avg_margin
                best_step = step
            if acc >= cfg.judge_warmup_overfit_warn_acc:
                print(f"[warmup][warn] eval_acc {acc:.3f} >= overfit threshold {cfg.judge_warmup_overfit_warn_acc:.3f}")
            if (
                acc >= cfg.judge_warmup_early_stop_min_acc
                and avg_margin >= cfg.judge_warmup_early_stop_min_margin
            ):
                print(
                    f"[warmup] early stop at step={step}, eval_acc={acc:.4f}, "
                    f"eval_avg_margin={avg_margin:.4f}"
                )
                early_stopped = True
                break

    # Save reusable Judge weights (for train.py --judge_warmup_mode reuse)
    torch.save(judge.state_dict(), out_dir / "judge_model.pt")
    # Save HF-style backbone + tokenizer for reproducibility / inspection.
    judge.transformer.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    with open(out_dir / "warmup_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "start_model_path": cfg.start_model_path,
                "output_dir": str(out_dir),
                "warmup_steps": cfg.warmup_steps,
                "judge_batch_size": cfg.judge_batch_size,
                "judge_warmup_lr": cfg.judge_warmup_lr if cfg.judge_warmup_lr > 0 else cfg.judge_lr * 0.1,
                "warmup_seed": cfg.warmup_seed,
                "eval_ratio": cfg.judge_warmup_eval_ratio,
                "eval_every": cfg.judge_warmup_eval_every,
                "eval_pairs": len(eval_triples),
                "best_eval_pairwise_acc": best_eval,
                "best_eval_avg_margin": best_eval_margin,
                "best_step": best_step,
                "judge_warmup_early_stop_min_margin": cfg.judge_warmup_early_stop_min_margin,
                "eval_scores_file": str(scores_path) if eval_triples else "",
                "early_stopped": early_stopped,
            },
            f,
            indent=2,
        )

    print(f"Saved judge warmup checkpoint to: {out_dir}")
    print(f"  - {out_dir / 'judge_model.pt'}")
    print(f"  - {out_dir / 'warmup_meta.json'}")
    print(f"  - {out_dir / 'judge_warmup_pairs.jsonl'}")
    if eval_triples:
        print(f"  - {scores_path}")


if __name__ == "__main__":
    main()
