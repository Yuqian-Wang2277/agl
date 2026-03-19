"""Two-stage vLLM rollout engine for Actor-Judge Phase II.

Stage 1: strategy generation   (few-shot context → <strategy>…</strategy>)
Stage 2: answer   generation   (strategy + question → <answer>…</answer>)
Stage 3: environment evaluation (answer vs gold → y ∈ {-1, 0, 1})

Key engineering decisions (from review rounds):

P1 — log_prob_old is NOT fetched from vLLM here.
     vLLM returns pure text only.  log_prob_old is computed by the Actor model
     itself inside actor_trainer.train_step() before the first mini-batch,
     using the same tokenizer as FSDP training — perfectly aligned, zero offset.

P2 — Ray @ray.remote(num_gpus=0):
     PyTorch FSDP already occupies all GPUs.  Telling Ray to request num_gpus=N
     makes it wait forever for N idle cards.  We instead set num_gpus=0 and let
     vLLM's tensor_parallel_size handle the hardware itself.

P3 — gpu_memory_utilization ≤ 0.75:
     PyTorch's CUDA Context pins ~1-1.5 GB/GPU that empty_cache() cannot free.
     0.9 / 0.85 will OOM the moment vLLM starts.  0.75 leaves enough headroom.

M1 — vLLM is launched ONLY on Rank 0 (the main process).
     The generated experiences are broadcast to all ranks afterward.
     See train.py for the broadcast orchestration.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import ray
from vllm import LLM, SamplingParams

from buffer import Experience, UCBBuffer
from data_loader import RolloutSample
from env import evaluate, compute_length_penalty
from prompts import (
    STRATEGY_CLOSE,
    STRATEGY_OPEN,
    apply_chat_template,
    build_answer_prompt,
    build_strategy_prompt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ray Actor (P2: num_gpus=0 — let vLLM own the hardware via tensor_parallel_size)
# ---------------------------------------------------------------------------

@ray.remote(num_gpus=0)
class VLLMActor:
    """vLLM inference server managed as a Ray Actor.

    Lifecycle:
      - Created by Rank 0 before each rollout epoch.
      - Killed by Rank 0 after rollout to free GPU memory before FSDP training.
      - Recreated at the next epoch (startup ~10-15 s, but eliminates all OOM risk).
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 8,
        gpu_memory_utilization: float = 0.75,  # P3: ≤0.75
    ) -> None:
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True,   # M5: few-shot context is shared → big speedup
            trust_remote_code=True,
        )

    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
    ) -> List[str]:
        """Run batched generation and return plain text outputs."""
        outputs = self.llm.generate(prompts, sampling_params)
        return [o.outputs[0].text for o in outputs]


# ---------------------------------------------------------------------------
# RolloutEngine (stateless, called on Rank 0 only)
# ---------------------------------------------------------------------------

class RolloutEngine:
    """Orchestrates two-stage vLLM rollout for a batch of RolloutSamples.

    Does NOT hold the vLLM LLM object itself — that lives inside VLLMActor.
    This class is stateless; the LLM is passed in on each call so that the
    Ray lifecycle (create / kill) is fully controlled by train.py.
    """

    def __init__(self, tokenizer, cfg) -> None:
        self.tokenizer = tokenizer
        self.cfg = cfg

    # ------------------------------------------------------------------

    def run(
        self,
        batch: List[RolloutSample],
        vllm_actor,          # VLLMActor Ray remote handle
        global_step: int = 0,
    ) -> List[Experience]:
        """Run a full two-stage rollout on *batch* and return experiences.

        For each of the B questions, K strategy samples are generated.
        Returns up to B*K Experience objects (fewer if some are format-invalid).
        """
        B = len(batch)
        K = self.cfg.K

        # ── Stage 1: Strategy generation ─────────────────────────────────
        stage1_prompts: List[str] = []
        for sample in batch:
            messages = build_strategy_prompt(sample.fewshot_examples)
            prompt_str = apply_chat_template(self.tokenizer, messages)
            # Repeat K times so vLLM receives one entry per (question, sample)
            stage1_prompts.extend([prompt_str] * K)

        sp_strategy = SamplingParams(
            temperature=self.cfg.rollout_temperature,
            max_tokens=self.cfg.strategy_max_tokens,
            stop=[STRATEGY_CLOSE],
            # Do NOT include logprobs — see P1: log_prob_old is computed by
            # the Actor model inside actor_trainer for perfect token alignment.
        )

        stage1_texts_raw: List[str] = ray.get(
            vllm_actor.generate.remote(stage1_prompts, sp_strategy)
        )

        # Manually append the stop word (vLLM strips it from the output)
        stage1_texts = [t + STRATEGY_CLOSE for t in stage1_texts_raw]

        # ── Stage 2: Answer generation (only for format-valid strategies) ─
        # Build stage-2 prompts; track which (b, k) pairs are format-valid
        stage2_prompts: List[str] = []
        stage2_indices: List[tuple[int, int]] = []   # (batch_idx, k_idx)

        for b_idx in range(B):
            for k_idx in range(K):
                s_text = stage1_texts[b_idx * K + k_idx]
                if STRATEGY_OPEN in s_text and STRATEGY_CLOSE in s_text:
                    messages = build_answer_prompt(
                        strategy=s_text,
                        question=batch[b_idx].question,
                    )
                    prompt_str = apply_chat_template(self.tokenizer, messages)
                    stage2_prompts.append(prompt_str)
                    stage2_indices.append((b_idx, k_idx))

        sp_answer = SamplingParams(
            temperature=self.cfg.answer_temperature,
            max_tokens=self.cfg.answer_max_tokens,
            stop=["</answer>"],
        )

        if stage2_prompts:
            stage2_texts_raw: List[str] = ray.get(
                vllm_actor.generate.remote(stage2_prompts, sp_answer)
            )
            stage2_texts = [t + "</answer>" for t in stage2_texts_raw]
        else:
            stage2_texts = []

        # Map back to (b_idx, k_idx) → answer text
        answer_map: dict[tuple[int, int], str] = {}
        for i, (b_idx, k_idx) in enumerate(stage2_indices):
            answer_map[(b_idx, k_idx)] = stage2_texts[i]

        # ── Stage 3: Env evaluation + Experience construction ─────────────
        experiences: List[Experience] = []

        for b_idx, sample in enumerate(batch):
            # Build the context text once for all K strategies of this question
            context_lines: List[str] = []
            for ex in sample.fewshot_examples:
                inp = ex.get("input", "")
                tgt = ex.get("target", "")
                if isinstance(tgt, list):
                    tgt = tgt[0] if tgt else ""
                context_lines.append(f"Q: {inp}  A: {tgt}")
            context_text = "\n".join(context_lines)

            for k_idx in range(K):
                s_text = stage1_texts[b_idx * K + k_idx]
                a_text = answer_map.get((b_idx, k_idx), "")

                outcome = evaluate(s_text, a_text, sample.answer_gold)

                # Optional Length Penalty (only when coeff > 0)
                len_pen = compute_length_penalty(
                    s_text,
                    coeff=self.cfg.length_penalty_coeff,
                    threshold=self.cfg.length_penalty_threshold,
                )
                # len_pen is incorporated into the sparse reward at actor_trainer level
                # We store it in outcome only as a float adjustment note; the integer
                # outcome field is kept clean for pairwise Judge training.
                # (Actor trainer reads raw outcome + len_pen separately.)

                exp = UCBBuffer.make_experience(
                    context_text=context_text,
                    question=sample.question,
                    strategy=s_text,
                    outcome=outcome,
                    timestamp=global_step,
                )
                experiences.append(exp)

        n_valid = sum(1 for e in experiences if e.outcome >= 0)
        n_correct = sum(1 for e in experiences if e.outcome == 1)
        logger.info(
            "Rollout step %d: B=%d K=%d → %d experiences "
            "(%d valid, %d correct, %.1f%% pass)",
            global_step, B, K, len(experiences),
            n_valid, n_correct,
            100.0 * n_correct / max(n_valid, 1),
        )

        return experiences
