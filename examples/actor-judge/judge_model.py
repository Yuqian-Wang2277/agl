"""JudgeModel — Qwen3 backbone with a scalar scoring head.

Architecture:
  - transformer: Qwen3 LM backbone (without lm_head)
  - scalar_head:  nn.Linear(hidden_size, 1) → raw logit

The model is initialised from an Actor HuggingFace checkpoint; the lm_head
is discarded and replaced with a freshly initialised scalar_head.

The <|judge|> special token is expected to be the last token in every input
sequence.  The Scalar Head attends to its hidden state, giving a stable
semantic anchor for scoring.

Initialisation notes (must be done in train.py before wrapping with FSDP):
  1. tokenizer.add_tokens(["<|judge|>"])
  2. judge_model.transformer.resize_token_embeddings(len(tokenizer))
  3. Warm-start the new embedding from <|im_end|> so the first few epochs
     don't produce pure-noise hidden states.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

logger = logging.getLogger(__name__)

JUDGE_TOKEN = "<|judge|>"


class JudgeModel(nn.Module):
    """Qwen3 backbone + scalar scoring head."""

    def __init__(self, transformer: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.transformer = transformer
        self.scalar_head = nn.Linear(hidden_size, 1, bias=True)
        # Small initialisation to keep early logits near 0
        nn.init.normal_(self.scalar_head.weight, std=0.02)
        nn.init.zeros_(self.scalar_head.bias)

    # ------------------------------------------------------------------

    @classmethod
    def from_actor_checkpoint(
        cls,
        checkpoint_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "JudgeModel":
        """Initialise JudgeModel from an Actor HuggingFace checkpoint.

        Loads the full CausalLM, extracts the backbone (model.model in Qwen3),
        and discards the lm_head.  The scalar_head is randomly initialised.
        """
        logger.info("Loading JudgeModel backbone from %s", checkpoint_path)
        causal_lm = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch_dtype,
            device_map="cpu",        # keep on CPU; accelerate will shard later
        )

        # Qwen3 / most HF LMs expose the backbone as .model
        if hasattr(causal_lm, "model"):
            backbone = causal_lm.model
        else:
            # Fallback: some architectures use .transformer
            backbone = causal_lm.transformer

        hidden_size = causal_lm.config.hidden_size

        del causal_lm   # free the lm_head memory immediately
        torch.cuda.empty_cache()

        judge = cls(backbone, hidden_size)
        logger.info(
            "JudgeModel created: hidden_size=%d, scalar_head=%s",
            hidden_size, judge.scalar_head,
        )
        return judge

    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score a batch of (context, question, strategy) inputs.

        Args:
            input_ids:      [B, seq_len]
            attention_mask: [B, seq_len]  (0 for PAD, 1 for real tokens)

        Returns:
            logits: [B]  — raw scalar logit per sample.
                    Apply sigmoid outside (e.g. in loss computation) so that
                    the L2 regularisation acts on the pre-sigmoid value.
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state   # [B, seq_len, hidden_size]

        # Locate the last *valid* token (= <|judge|> anchor) in each sequence.
        # DO NOT use hidden[:, -1, :] — that would pick the PAD token when
        # sequences have been right-padded.
        seq_lens  = attention_mask.sum(dim=1) - 1           # [B]
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        last_hidden = hidden[batch_idx, seq_lens, :]         # [B, hidden_size]

        logit = self.scalar_head(last_hidden).squeeze(-1)    # [B]
        return logit


# ---------------------------------------------------------------------------
# Warm-start helper (called once in train.py after resize_token_embeddings)
# ---------------------------------------------------------------------------

def warmstart_judge_token_embedding(
    judge_model: JudgeModel,
    tokenizer,
    source_token: str = "<|im_end|>",
) -> None:
    """Copy the embedding of *source_token* into the newly added <|judge|> slot.

    After resize_token_embeddings the new token gets a random vector.  That
    random vector, passed through 30+ Transformer layers, outputs pure noise.
    The Judge will be extremely confused for the first few epochs, causing BT
    Loss to spike.  Initialising from a semantically similar token (e.g.
    <|im_end|>, the Qwen3 end-of-turn marker) gives a sane starting point.
    """
    src_id = tokenizer.convert_tokens_to_ids(source_token)
    if src_id == tokenizer.unk_token_id:
        logger.warning(
            "warmstart_judge_token_embedding: source token '%s' not found in "
            "tokenizer vocabulary — skipping warm-start.",
            source_token,
        )
        return

    emb_table = judge_model.transformer.get_input_embeddings()
    with torch.no_grad():
        emb_table.weight[-1] = emb_table.weight[src_id].clone()

    logger.info(
        "Warm-started <|judge|> embedding from '%s' (id=%d).",
        source_token, src_id,
    )
