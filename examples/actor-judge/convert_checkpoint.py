"""Convert VERL FSDP sharded checkpoint → HuggingFace safetensors format.

The strategy_extraction training (Phase I) uses VERL's FSDP backend, which
saves 8 rank-sharded .pt files.  This script merges them into a single
HuggingFace-compatible directory so Actor, Ref, and Judge can be loaded with
AutoModelForCausalLM.from_pretrained().

Usage:
    python convert_checkpoint.py \
        --checkpoint_dir /path/to/global_step_400/actor \
        --output_dir     /path/to/global_step_400/actor_hf

The script tries accelerate's merge_fsdp_weights first (cleanest).
If that API is unavailable (version mismatch), it falls back to a manual
torch.load + state_dict merge routine.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Method 1: accelerate.utils.merge_fsdp_weights  (preferred)
# ---------------------------------------------------------------------------

def convert_via_accelerate(checkpoint_dir: str, output_dir: str) -> bool:
    """Try to merge using accelerate's built-in FSDP consolidation API.

    Returns True on success, False if the API is unavailable.
    """
    try:
        from accelerate.utils import merge_fsdp_weights   # accelerate ≥ 0.26
    except ImportError:
        logger.warning("accelerate.utils.merge_fsdp_weights not found — trying fallback.")
        return False

    logger.info("Merging FSDP shards via accelerate.utils.merge_fsdp_weights …")
    try:
        merge_fsdp_weights(
            checkpoint_dir=checkpoint_dir,
            output_path=output_dir,
            safe_serialization=True,
        )
        logger.info("Merged successfully → %s", output_dir)
        return True
    except Exception as exc:
        logger.warning("merge_fsdp_weights raised: %s — trying fallback.", exc)
        return False


# ---------------------------------------------------------------------------
# Method 2: Manual shard merge
# ---------------------------------------------------------------------------

def _find_shard_files(checkpoint_dir: str) -> list[str]:
    """Return sorted list of model_world_size_*_rank_*.pt shard paths."""
    ckpt_path = Path(checkpoint_dir)
    shards = sorted(ckpt_path.glob("model_world_size_*_rank_*.pt"))
    if not shards:
        # Some VERL versions use a different naming
        shards = sorted(ckpt_path.glob("*.pt"))
    return [str(s) for s in shards]


def convert_manual(checkpoint_dir: str, output_dir: str) -> None:
    """Manually merge FSDP shards via torch.load + average/concatenate."""
    shards = _find_shard_files(checkpoint_dir)
    if not shards:
        raise FileNotFoundError(
            f"No shard .pt files found in {checkpoint_dir}"
        )
    logger.info("Found %d shard(s): %s …", len(shards), shards[:2])

    merged: dict[str, torch.Tensor] = {}
    for shard_path in shards:
        logger.info("  Loading shard %s", shard_path)
        state = torch.load(shard_path, map_location="cpu")
        if isinstance(state, dict) and "module" in state:
            state = state["module"]   # VERL wraps the model under 'module'
        for k, v in state.items():
            if k not in merged:
                merged[k] = v.clone()
            else:
                # For FSDP flat parameters, shards are concatenated not summed
                merged[k] = torch.cat([merged[k], v], dim=0)

    logger.info("Merged %d parameter tensors.", len(merged))

    os.makedirs(output_dir, exist_ok=True)
    # Save in safetensors format if available
    try:
        from safetensors.torch import save_file
        save_file(merged, os.path.join(output_dir, "model.safetensors"))
        logger.info("Saved as safetensors → %s/model.safetensors", output_dir)
    except ImportError:
        out_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(merged, out_path)
        logger.info("Saved as pytorch_model.bin → %s", out_path)


# ---------------------------------------------------------------------------
# Tokenizer copy
# ---------------------------------------------------------------------------

def copy_tokenizer(checkpoint_dir: str, output_dir: str) -> None:
    """Copy tokenizer files from checkpoint_dir/huggingface/ to output_dir."""
    hf_dir = Path(checkpoint_dir) / "huggingface"
    if not hf_dir.exists():
        logger.warning("No huggingface/ subdir found — tokenizer not copied.")
        return

    os.makedirs(output_dir, exist_ok=True)
    for f in hf_dir.iterdir():
        dst = Path(output_dir) / f.name
        shutil.copy2(str(f), str(dst))
        logger.info("  Copied tokenizer file: %s", f.name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert VERL FSDP checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Path to the VERL actor checkpoint directory (e.g. global_step_400/actor)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Destination directory for the HuggingFace-format model",
    )
    args = parser.parse_args()

    # Try accelerate first, fall back to manual merge
    success = convert_via_accelerate(args.checkpoint_dir, args.output_dir)
    if not success:
        logger.info("Falling back to manual shard merge …")
        convert_manual(args.checkpoint_dir, args.output_dir)

    # Always copy tokenizer files
    copy_tokenizer(args.checkpoint_dir, args.output_dir)

    # Quick sanity check
    out = Path(args.output_dir)
    files = list(out.iterdir())
    logger.info(
        "Output directory contains %d files: %s",
        len(files),
        [f.name for f in files[:5]],
    )
    logger.info("Done.  Load with: AutoModelForCausalLM.from_pretrained('%s')", args.output_dir)


if __name__ == "__main__":
    main()
