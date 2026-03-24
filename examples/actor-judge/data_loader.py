"""Data loading for Actor-Judge Phase II training.

Provides:
  - RolloutSample: typed dataclass per training item
  - ActorJudgeDataset: PyTorch Dataset wrapping LLMReflection JSON data
  - load_rollout_dataset / load_val_dataset: convenience factory functions
  - build_stratified_rollout_partitions: domain-stratified disjoint Phase-A batch groups
  - match_s_gold / match_s_gold_versions: find gold strategy files for Judge warmup
"""

from __future__ import annotations

import glob
import json
import logging
import os
import random
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import Dataset

from prompts import apply_chat_template, build_strategy_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RolloutSample:
    """One training item fed to the rollout engine."""
    domain: str                         # task / problem-type name
    fewshot_examples: List[Dict[str, Any]]  # k examples for strategy generation
    question: str                       # new question Q' the actor must answer
    answer_gold: str                    # ground-truth answer
    s_gold: Optional[str] = None        # gold strategy (only used for warmup)
    s_gold_by_version: Optional[Dict[str, str]] = None   # e.g. {"v1": "...", "v2": "..."}


# ---------------------------------------------------------------------------
# Low-level JSON loading (compatible with strategy_extraction format)
# ---------------------------------------------------------------------------

def _load_domain_examples(data_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load all problem types from a LLMReflection data directory.

    Returns dict mapping domain-name → list of {"input": str, "target": list[str]}.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    domain_map: Dict[str, List[Dict[str, Any]]] = {}

    for domain_dir in sorted(data_path.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name
        examples: List[Dict[str, Any]] = []

        for json_file in sorted(domain_dir.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "examples" in data:
                    examples.extend(data["examples"])
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Skipping %s: %s", json_file, exc)

        if examples:
            domain_map[domain] = examples

    logger.info("Loaded %d domains from %s", len(domain_map), data_dir)
    return domain_map


def _pick_answer(target: Any) -> str:
    """Normalise 'target' field (list or string) to a plain string."""
    if isinstance(target, list):
        return str(target[0]) if target else ""
    return str(target)


# ---------------------------------------------------------------------------
# S_gold matching
# ---------------------------------------------------------------------------

def match_s_gold_versions(domain: str, strategy_dir: str) -> Dict[str, str]:
    """Find gold strategies for all known prompt formats (v1/v2/v3).

    Returns a dict like {"v1": "...", "v2": "...", "v3": "..."} with only
    available keys populated.
    """
    base = Path(strategy_dir) / "train_all" / "gain_pos"
    if not base.exists():
        return {}

    domain_norm = _normalise_domain(domain)
    version_to_text: Dict[str, str] = {}
    version_to_dir_glob = {
        "v1": "useful_by_coarse*_v1",  # handles both useful_by_coarse_v1 and typo variants
        "v2": "useful_by_coarse*_v2",
        "v3": "useful_by_coarse*_v3",
    }

    for version, version_dir in version_to_dir_glob.items():
        pattern = str(
            base / version_dir / "**" / "strategies_out" / "size_*" / "001.cand1.strategy.txt"
        )
        candidates = sorted(glob.glob(pattern, recursive=True))
        for path in candidates:
            path_norm = _normalise_domain(path)
            if domain_norm in path_norm:
                try:
                    version_to_text[version] = Path(path).read_text(encoding="utf-8").strip()
                    break
                except OSError:
                    continue
    return version_to_text


def match_s_gold(domain: str, strategy_dir: str) -> Optional[str]:
    """Return one default gold strategy (prefers v1→v2→v3)."""
    by_version = match_s_gold_versions(domain, strategy_dir)
    for version in ("v1", "v2", "v3"):
        if version in by_version:
            return by_version[version]
    return None


def _normalise_domain(s: str) -> str:
    """Lower-case and replace non-alphanumeric chars with underscores."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower())


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ActorJudgeDataset(Dataset):
    """PyTorch Dataset of RolloutSample objects.

    Samples are created eagerly at construction time for reproducibility.
    Same-domain and cross-domain samples are interleaved according to
    *cross_domain_ratio*.
    """

    def __init__(
        self,
        data_dir: str,
        fewshot_min: int = 3,
        fewshot_max: int = 5,
        num_samples: int = 20_000,
        cross_domain_ratio: float = 0.5,
        strategy_dir: str = "",
        load_s_gold: bool = False,
        seed: int = 42,
        *,
        max_stage1_prompt_tokens: int = 0,
        tokenizer: Optional[Any] = None,
        stage1_chars_per_token: float = 2.5,
        stage1_reject_log_path: str = "",
    ) -> None:
        random.seed(seed)
        self._domain_map = _load_domain_examples(data_dir)
        self._domains = sorted(self._domain_map.keys())
        if len(self._domains) < 2:
            raise ValueError(
                f"Need at least 2 domains for cross-domain sampling, "
                f"found {len(self._domains)} in {data_dir}"
            )

        self._fewshot_min = fewshot_min
        self._fewshot_max = fewshot_max
        self._cross_domain_ratio = cross_domain_ratio
        self._strategy_dir = strategy_dir
        self._load_s_gold = load_s_gold
        self._max_stage1 = int(max_stage1_prompt_tokens)
        self._tokenizer = tokenizer
        self._stage1_cpt = float(stage1_chars_per_token)
        self._reject_log_path = (stage1_reject_log_path or "").strip()
        self._n_rejected_long = 0

        # Pre-cache gold strategies to avoid repeated glob scans
        self._s_gold_cache: Dict[str, Optional[str]] = {}
        self._s_gold_versions_cache: Dict[str, Dict[str, str]] = {}

        self._samples: List[RolloutSample] = self._build(num_samples)

    # ------------------------------------------------------------------

    def _build(self, num_samples: int) -> List[RolloutSample]:
        samples: List[RolloutSample] = []
        n_cross = int(num_samples * self._cross_domain_ratio)
        n_same = num_samples - n_cross

        self._fill(samples, n_same, cross_domain=False)
        self._fill(samples, n_cross, cross_domain=True)

        random.shuffle(samples)
        logger.info(
            "Dataset built: %d samples (%d same-domain target, %d cross-domain target; "
            "%d rejected for long Stage-1 prompt)",
            len(samples),
            n_same,
            n_cross,
            self._n_rejected_long,
        )
        return samples

    def _stage1_prompt_token_len(self, sample: RolloutSample) -> float:
        """Tokens for the Stage-1 chat string (few-shot + template); no strategy yet."""
        messages = build_strategy_prompt(sample.fewshot_examples)
        if self._tokenizer is not None:
            prompt_str = apply_chat_template(self._tokenizer, messages)
            return float(
                len(
                    self._tokenizer.encode(
                        prompt_str,
                        add_special_tokens=False,
                    )
                )
            )
        flat = "\n".join(m.get("content", "") for m in messages)
        return len(flat) / max(self._stage1_cpt, 1e-6)

    def _reject_long(self, sample: RolloutSample, est_tokens: float, cross_domain: bool) -> None:
        self._n_rejected_long += 1
        q_prev = sample.question[:200] + ("…" if len(sample.question) > 200 else "")
        logger.warning(
            "Stage-1 length gate: rejected est_tokens=%.1f (threshold=%d) domain=%s "
            "cross_domain=%s question_preview=%r",
            est_tokens,
            self._max_stage1,
            sample.domain,
            cross_domain,
            q_prev,
        )
        if self._reject_log_path:
            Path(self._reject_log_path).parent.mkdir(parents=True, exist_ok=True)
            rec = {
                "reason": "stage1_prompt_too_long",
                "est_tokens": round(est_tokens, 2),
                "threshold": self._max_stage1,
                "domain": sample.domain,
                "cross_domain": cross_domain,
                "n_fewshot": len(sample.fewshot_examples),
                "question": sample.question,
                "answer_gold": sample.answer_gold,
            }
            with open(self._reject_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _fill(self, samples: List[RolloutSample], target: int, cross_domain: bool) -> None:
        if target <= 0:
            return
        added = 0
        attempts = 0
        max_attempts = max(target * 200, 10_000)
        while added < target and attempts < max_attempts:
            attempts += 1
            s = self._make_sample(cross_domain=cross_domain)
            if s is None:
                continue
            est = self._stage1_prompt_token_len(s)
            if self._max_stage1 <= 0 or est <= self._max_stage1:
                samples.append(s)
                added += 1
            else:
                self._reject_long(s, est, cross_domain)
        if added < target:
            logger.warning(
                "ActorJudgeDataset: got %d/%d %s samples after %d attempts "
                "(length gate, sparse pools, or cross-domain constraints).",
                added,
                target,
                "cross-domain" if cross_domain else "same-domain",
                attempts,
            )

    def _make_sample(self, cross_domain: bool) -> Optional[RolloutSample]:
        """Build one RolloutSample.

        For same-domain: fewshot context and question come from the same domain.
        For cross-domain: fewshot context comes from domain D1, question from D2 ≠ D1.
        """
        domain_fs = random.choice(self._domains)
        if cross_domain:
            others = [d for d in self._domains if d != domain_fs]
            domain_q = random.choice(others)
        else:
            domain_q = domain_fs

        fs_pool = self._domain_map[domain_fs]
        q_pool  = self._domain_map[domain_q]

        n_shots = random.randint(self._fewshot_min, self._fewshot_max)
        if len(fs_pool) < n_shots + 1 or len(q_pool) < 1:
            return None

        fewshot_idxs = random.sample(range(len(fs_pool)), n_shots)
        fewshot = [fs_pool[i] for i in fewshot_idxs]

        # Choose question that is NOT in the fewshot pool (when same domain)
        q_candidates = (
            [q_pool[i] for i in range(len(q_pool)) if i not in fewshot_idxs]
            if not cross_domain else list(q_pool)
        )
        if not q_candidates:
            return None
        q_example = random.choice(q_candidates)

        s_gold_by_version = self._get_s_gold_versions(domain_fs) if self._load_s_gold else None
        s_gold = self._get_s_gold(domain_fs) if self._load_s_gold else None

        return RolloutSample(
            domain=domain_fs,
            fewshot_examples=fewshot,
            question=q_example.get("input", ""),
            answer_gold=_pick_answer(q_example.get("target", "")),
            s_gold=s_gold,
            s_gold_by_version=s_gold_by_version,
        )

    def _get_s_gold(self, domain: str) -> Optional[str]:
        if domain not in self._s_gold_cache:
            self._s_gold_cache[domain] = (
                match_s_gold(domain, self._strategy_dir)
                if self._strategy_dir else None
            )
        return self._s_gold_cache[domain]

    def _get_s_gold_versions(self, domain: str) -> Dict[str, str]:
        if domain not in self._s_gold_versions_cache:
            self._s_gold_versions_cache[domain] = (
                match_s_gold_versions(domain, self._strategy_dir)
                if self._strategy_dir else {}
            )
        return self._s_gold_versions_cache[domain]

    # ------------------------------------------------------------------
    # Dataset interface

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> RolloutSample:
        return self._samples[idx]


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------

def build_stratified_rollout_partitions(
    dataset: "ActorJudgeDataset",
    batch_size: int,
    rollout_limit: int,
    seed: int,
) -> Tuple[int, List[List[int]]]:
    """Split all rollout batch indices into disjoint groups of size ``rollout_limit``.

    Each batch is labeled by the **first** sample's ``domain``. Batches are shuffled
    within each domain, then **round-robin merged** across domains (so each chunk is
    domain-mixed rather than one domain dominating a partition). The merged list is
    cut into contiguous segments of length ``rollout_limit``.

    The previous implementation split each domain separately across ``n_groups`` buckets;
    that does **not** guarantee equal bucket totals when domains are imbalanced, which
    caused ``RuntimeError: group g has != rollout_limit batches``.

    Requires:
      - ``len(dataset) % batch_size == 0``
      - ``(len(dataset) // batch_size) % rollout_limit == 0``

    Returns:
      ``(n_groups, partitions)`` where ``partitions[k]`` lists batch indices for epoch
      ``k mod n_groups`` — each batch index appears exactly once across all partitions.
    """
    n = len(dataset)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if rollout_limit <= 0:
        raise ValueError(f"rollout_limit must be > 0 for stratified partitions, got {rollout_limit}")
    if n % batch_size != 0:
        raise ValueError(
            f"Stratified rollout requires len(dataset)={n} divisible by batch_size={batch_size}"
        )
    n_batches = n // batch_size
    if n_batches % rollout_limit != 0:
        raise ValueError(
            f"Stratified rollout requires (len(dataset)//batch_size)={n_batches} "
            f"divisible by rollout_steps_per_epoch={rollout_limit} "
            f"(full coverage in n_batches/rollout_limit epochs with no duplicate batches)"
        )
    n_groups = n_batches // rollout_limit
    rng = random.Random(seed)

    by_domain: Dict[str, List[int]] = defaultdict(list)
    for bi in range(n_batches):
        s0 = dataset[bi * batch_size]
        by_domain[s0.domain].append(bi)

    for dom in by_domain:
        rng.shuffle(by_domain[dom])

    # Round-robin across domains so merged order mixes strata; then equal-size chunks.
    queues: List[deque[int]] = [deque(by_domain[k]) for k in sorted(by_domain.keys())]
    merged: List[int] = []
    while any(queues):
        for q in queues:
            if q:
                merged.append(q.popleft())

    if len(merged) != n_batches:
        raise RuntimeError(f"internal: merged length {len(merged)} != n_batches {n_batches}")

    groups = [
        merged[i : i + rollout_limit] for i in range(0, n_batches, rollout_limit)
    ]
    if len(groups) != n_groups:
        raise RuntimeError(f"internal: {len(groups)} groups != n_groups {n_groups}")

    for g in groups:
        rng.shuffle(g)

    return n_groups, groups


def load_rollout_dataset(
    cfg,
    tokenizer: Optional[Any] = None,
    *,
    stage1_reject_log_path: Optional[str] = None,
    max_stage1_prompt_tokens: Optional[int] = None,
) -> ActorJudgeDataset:
    """Build the training dataset from an ActorJudgeConfig."""
    train_dir = os.path.join(cfg.data_base_path, cfg.train_subdir)
    if max_stage1_prompt_tokens is not None:
        mx = int(max_stage1_prompt_tokens)
    else:
        mx = int(getattr(cfg, "max_stage1_prompt_tokens", 0))
    log_path = ""
    if mx > 0 and getattr(cfg, "dataset_stage1_reject_log", True):
        if stage1_reject_log_path:
            log_path = str(Path(stage1_reject_log_path).expanduser())
        else:
            cd = (getattr(cfg, "checkpoint_dir", None) or "").strip()
            if cd:
                log_path = str(Path(cd) / "dataset_stage1_rejects.jsonl")
    return ActorJudgeDataset(
        data_dir=train_dir,
        fewshot_min=cfg.fewshot_min,
        fewshot_max=cfg.fewshot_max,
        num_samples=getattr(cfg, 'num_train_samples', 20_000),   # L2: configurable
        cross_domain_ratio=cfg.cross_domain_ratio,
        strategy_dir=cfg.strategy_dir,
        load_s_gold=cfg.judge_warmup,
        max_stage1_prompt_tokens=mx,
        tokenizer=tokenizer,
        stage1_chars_per_token=float(getattr(cfg, "stage1_length_chars_per_token", 2.5)),
        stage1_reject_log_path=log_path,
    )


def load_val_dataset(
    cfg,
    subdir: str,
    tokenizer: Optional[Any] = None,
    *,
    stage1_reject_log_path: Optional[str] = None,
    max_stage1_prompt_tokens: Optional[int] = None,
) -> ActorJudgeDataset:
    """Build a validation dataset for one val split."""
    val_dir = os.path.join(cfg.data_base_path, subdir)
    n_val = getattr(cfg, "val_num_samples", 500)
    if max_stage1_prompt_tokens is not None:
        mx = int(max_stage1_prompt_tokens)
    else:
        mx = int(getattr(cfg, "max_stage1_prompt_tokens", 0))
    log_path = ""
    if mx > 0 and getattr(cfg, "dataset_stage1_reject_log", True):
        if stage1_reject_log_path:
            log_path = str(Path(stage1_reject_log_path).expanduser())
        else:
            cd = (getattr(cfg, "checkpoint_dir", None) or "").strip()
            if cd:
                log_path = str(Path(cd) / f"dataset_stage1_rejects_val_{subdir}.jsonl")
    return ActorJudgeDataset(
        data_dir=val_dir,
        fewshot_min=cfg.fewshot_min,
        fewshot_max=cfg.fewshot_max,
        num_samples=n_val,
        cross_domain_ratio=0.0,   # val is always same-domain
        load_s_gold=False,
        max_stage1_prompt_tokens=mx,
        tokenizer=tokenizer,
        stage1_chars_per_token=float(getattr(cfg, "stage1_length_chars_per_token", 2.5)),
        stage1_reject_log_path=log_path,
    )


def collate_rollout_samples(
    samples: List[RolloutSample],
) -> Tuple[List[RolloutSample], List[str]]:
    """Identity collate — keep RolloutSamples as a plain list.

    The rollout engine and trainers expect raw Python objects, not tensors,
    so we intentionally skip tensor collation here.
    """
    return samples, [s.domain for s in samples]
