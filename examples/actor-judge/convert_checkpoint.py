"""Convert VERL FSDP sharded checkpoint → HuggingFace safetensors format.

The strategy_extraction training (Phase I) uses VERL's FSDP backend, which
saves 8 rank-sharded .pt files.  This script merges them into a single
HuggingFace-compatible directory so Actor, Ref, and Judge can be loaded with
AutoModelForCausalLM.from_pretrained().

Usage:
    python convert_checkpoint.py \
        --checkpoint_dir /path/to/global_step_400/actor \
        --output_dir     /path/to/global_step_400/actor_hf

The script tries ``accelerate.utils.merge_fsdp_weights`` first.  That API
expects a **PyTorch Distributed Checkpoint** layout (``.metadata`` under
``checkpoint_dir``).  VERL instead saves **per-rank** ``model_world_size_*_rank_*.pt``
files containing ``DTensor`` shards, so merge_fsdp_weights usually fails and we
fall back to a manual ``to_local()`` + ``torch.cat`` merge.

If merge_fsdp_weights raises ``CheckpointException``, note that in recent PyTorch
it subclasses ``BaseException`` (not ``Exception``); we catch ``BaseException``
so the manual path actually runs.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

import torch

# Register DTensor unpickler when checkpoints were saved with torch.save(DTensor, …).
import torch.distributed.tensor  # noqa: F401

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
    except BaseException as exc:
        # PyTorch's CheckpointException subclasses BaseException, not Exception —
        # so a bare ``except Exception`` never runs the manual merge fallback.
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
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
        # Other layouts: only pick model shards, never optim_*.pt / extra_state_*.pt
        shards = sorted(
            p for p in ckpt_path.glob("*.pt") if p.name.startswith("model_")
        )
    return [str(s) for s in shards]


def _local_cpu_tensor(v: object) -> torch.Tensor:
    """Turn a shard value into a plain CPU tensor for concatenation.

    VERL FSDP checkpoints often store ``DTensor`` objects.  ``torch.cat`` on two
    DTensors tries to run collective ops and needs a process group; we instead
    take each rank's ``to_local()`` slice and cat those on CPU.
    """
    import torch.distributed.tensor as dist_tensor

    if isinstance(v, dist_tensor.DTensor):
        return v.to_local().detach().cpu().clone()
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().clone()
    raise TypeError(f"Unexpected state_dict value type {type(v)!r} (expected Tensor/DTensor)")


def convert_manual(checkpoint_dir: str, output_dir: str) -> None:
    """Manually merge FSDP shards via torch.load + concat along dim 0."""
    shards = _find_shard_files(checkpoint_dir)
    if not shards:
        raise FileNotFoundError(
            f"No shard .pt files found in {checkpoint_dir}"
        )
    logger.info("Found %d shard(s): %s …", len(shards), shards[:2])

    merged: dict[str, torch.Tensor] = {}
    for shard_path in shards:
        logger.info("  Loading shard %s", shard_path)
        state = torch.load(shard_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "module" in state:
            state = state["module"]   # VERL wraps the model under 'module'
        for k, v in state.items():
            piece = _local_cpu_tensor(v)
            if k not in merged:
                merged[k] = piece
            else:
                # Sharded rows / first-dim chunks (FSDP on dim 0)
                merged[k] = torch.cat([merged[k], piece], dim=0)

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

def _copy_file_robust(src: Path, dst: Path) -> None:
    """Copy bytes without sendfile (some NFS / FUSE mounts break os.sendfile)."""
    with open(src, "rb") as fsrc:
        data = fsrc.read()
    with open(dst, "wb") as fdst:
        fdst.write(data)


def copy_tokenizer(checkpoint_dir: str, output_dir: str) -> None:
    """Copy tokenizer files from checkpoint_dir/huggingface/ to output_dir."""
    hf_dir = Path(checkpoint_dir) / "huggingface"
    if not hf_dir.exists():
        logger.warning("No huggingface/ subdir found — tokenizer not copied.")
        return

    os.makedirs(output_dir, exist_ok=True)
    for f in sorted(hf_dir.iterdir(), key=lambda p: p.name):
        dst = Path(output_dir) / f.name
        try:
            shutil.copy2(str(f), str(dst))
        except OSError as exc:
            # copy2/copyfile may both use sendfile; fall back to read/write
            logger.warning("copy2 failed for %s (%s); retrying with read/write.", f.name, exc)
            _copy_file_robust(f, dst)
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
