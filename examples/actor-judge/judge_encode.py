"""Judge input encoding with guaranteed <|judge|> anchor as the last real token.

HF ``truncation=True`` on the full string can drop the trailing anchor; we
truncate the body only, then append anchor ids.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from transformers import PreTrainedTokenizer

from prompts import JUDGE_TOKEN


def judge_anchor_token_ids(tokenizer: PreTrainedTokenizer) -> List[int]:
    ids = tokenizer.encode(JUDGE_TOKEN, add_special_tokens=False)
    if not ids:
        raise RuntimeError(
            f"Tokenizer returned no ids for {JUDGE_TOKEN!r}; add the special token first."
        )
    return ids


def encode_batch_for_judge(
    tokenizer: PreTrainedTokenizer,
    body_texts: List[str],
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """Batch-encode judge bodies; each row ends with anchor token ids, length ≤ max_length."""
    if not body_texts:
        return {
            "input_ids": torch.zeros((0, 0), dtype=torch.long),
            "attention_mask": torch.zeros((0, 0), dtype=torch.long),
        }
    anchor = judge_anchor_token_ids(tokenizer)
    reserve = len(anchor)
    content_max = max(1, max_length - reserve)
    pad_id = tokenizer.pad_token_id or 0

    rows_ids: List[List[int]] = []
    for body in body_texts:
        ids = tokenizer.encode(
            (body or "").strip(),
            add_special_tokens=False,
            truncation=True,
            max_length=content_max,
        )
        full = ids + anchor
        if len(full) > max_length:
            overflow = len(full) - max_length
            ids = ids[overflow:]
            full = ids + anchor
        rows_ids.append(full)

    max_len = min(max_length, max(len(r) for r in rows_ids))
    bsz = len(rows_ids)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, ids in enumerate(rows_ids):
        row = ids[:max_len]
        L = len(row)
        input_ids[i, :L] = torch.tensor(row, dtype=torch.long)
        attention_mask[i, :L] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}
