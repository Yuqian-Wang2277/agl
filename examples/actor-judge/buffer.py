"""UCB Reflective Experience Replay Buffer.

Design decisions:
- Two-level dict:  _data[q_hash][traj_id] = Experience
  Gives O(1) add, O(1) v_pred write-back, and O(1) per-Q eviction.
- Per-Q capacity (per_q_max): prevents hard questions (always y=0) from
  flooding the buffer.  When the limit is exceeded the trajectory with the
  lowest UCB score is evicted.
- global_sample_steps is the T in the UCB formula; it increments every time
  sample_pairwise() is called — NOT on every add().
- Experience stores ONLY pure-Python scalars (str/int/float/list-of-dicts).
  No torch.Tensors!  If a Tensor ended up here, broadcast_object_list would
  trigger an NCCL hang with no error message.

Fields added since original design:
  prompt_text   — full chat-template formatted Stage-1 prompt string.
                  Used by actor_trainer to compute log_prob over the EXACT
                  token sequence that vLLM generated against.  Without this,
                  the prompt reconstructed in actor_trainer would differ from
                  vLLM's actual input, making action_mask boundaries wrong.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experience dataclass — all fields MUST be pure Python (no GPU tensors)
# ---------------------------------------------------------------------------

@dataclass
class Experience:
    traj_id: str        # UUID string → O(1) lookup for v_pred write-back
    context_text: str   # few-shot examples as "Q: A:" plain text (for Judge prompt)
    prompt_text: str    # full chat-template formatted Stage-1 prompt (for actor log_prob)
    question: str       # new question Q'
    strategy: str       # generated strategy S (full text including tags)
    outcome: int        # y ∈ {-1, 0, 1}; -1 = format error, not used in pairwise
    n_sampled: int = 0       # how many times this traj was sampled by Judge
    v_pred: float = 0.5      # Judge's latest predicted score (initialised at 0.5)
    timestamp: int = 0       # global step when this experience was added


# ---------------------------------------------------------------------------
# Pairwise pair returned by sample_pairwise
# ---------------------------------------------------------------------------

class Pair(NamedTuple):
    q_hash: str
    traj_id_win: str
    traj_id_lose: str
    exp_win: Experience
    exp_lose: Experience


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------

def _hash_question(question: str) -> str:
    return hashlib.md5(question.encode("utf-8")).hexdigest()


class UCBBuffer:
    """UCB-prioritised experience replay buffer.

    Args:
        max_size:     Global upper bound on number of stored experiences.
        per_q_max:    Per-question upper bound (evict lowest-UCB on overflow).
        lambda_err:   Weight for exploitation term  |y - v_pred|.
        lambda_exp:   Weight for exploration term   √(ln T / (n + 1)).
        disable_ucb:  If True, fall back to FIFO uniform sampling (ablation E).
    """

    def __init__(
        self,
        max_size: int = 50_000,
        per_q_max: int = 50,
        lambda_err: float = 1.0,
        lambda_exp: float = 1.0,
        disable_ucb: bool = False,
    ) -> None:
        # _data[q_hash][traj_id] = Experience
        self._data: Dict[str, Dict[str, Experience]] = {}
        self.global_sample_steps: int = 0
        self.max_size = max_size
        self.per_q_max = per_q_max
        self.lambda_err = lambda_err
        self.lambda_exp = lambda_exp
        self.disable_ucb = disable_ucb
        self._total: int = 0  # fast size counter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, exp: Experience) -> None:
        """Insert one experience, enforcing per-Q and global capacity limits."""
        q_hash = _hash_question(exp.question)

        if q_hash not in self._data:
            self._data[q_hash] = {}

        # Evict globally if at capacity (oldest-first by timestamp)
        if self._total >= self.max_size:
            self._evict_oldest_global()

        self._data[q_hash][exp.traj_id] = exp
        self._total += 1

        # Per-Q capacity: evict the trajectory with the lowest UCB score
        if len(self._data[q_hash]) > self.per_q_max:
            self._evict_worst_in_q(q_hash)

    def update_v_pred(self, q_hash: str, traj_id: str, new_v: float) -> None:
        """O(1) write-back of Judge's predicted score."""
        try:
            exp = self._data[q_hash][traj_id]
            exp.v_pred = new_v
            exp.n_sampled += 1
        except KeyError:
            logger.debug("update_v_pred: traj %s not found (already evicted?)", traj_id)

    def sample_pairwise(self, batch_size: int) -> List[Pair]:
        """Sample *batch_size* (win, lose) pairs for Judge training.

        Only q_hashes that have at least one y=1 AND one y=0 experience are
        eligible.  y=-1 experiences (format errors) are always excluded.

        Selection is weighted by number of trajectories per q_hash so that
        questions with more stored trajectories contribute more to training
        (M6 fix: previously uniform, which under-represented rich q_hashes).

        The global_sample_steps counter (T in the UCB formula) is incremented
        once per call — not once per pair — so T tracks "judge update steps".

        Returns fewer than batch_size pairs if the buffer is too sparse.
        """
        self.global_sample_steps += 1
        # M1: use max(T, e) so log is always positive, even at T=1
        T = max(self.global_sample_steps, 1)

        eligible: List[str] = []
        eligible_weights: List[int] = []
        unipolar: int = 0

        for q_hash, trajs in self._data.items():
            wins  = [e for e in trajs.values() if e.outcome == 1]
            loses = [e for e in trajs.values() if e.outcome == 0]
            if wins and loses:
                eligible.append(q_hash)
                eligible_weights.append(len(trajs))   # M6: weight by size
            elif trajs:
                unipolar += 1

        # Warn if too many unipolar questions — suggests temperature too low
        total_q = len(self._data)
        if total_q > 0 and unipolar / total_q > 0.30:
            logger.warning(
                "%.0f%% of questions are unipolar (all-win or all-lose). "
                "Consider raising rollout_temperature.",
                100.0 * unipolar / total_q,
            )

        if not eligible:
            logger.debug("sample_pairwise: no eligible (win+lose) q_hashes yet.")
            return []

        pairs: List[Pair] = []
        for _ in range(batch_size):
            if not eligible:
                break
            # M6: weighted selection so questions with more trajectories contribute more
            q_hash = random.choices(eligible, weights=eligible_weights, k=1)[0]
            trajs  = self._data[q_hash]
            wins   = [e for e in trajs.values() if e.outcome == 1]
            loses  = [e for e in trajs.values() if e.outcome == 0]

            if self.disable_ucb:
                win_exp  = random.choice(wins)
                lose_exp = random.choice(loses)
            else:
                win_exp  = self._ucb_sample(wins,  T)
                lose_exp = self._ucb_sample(loses, T)

            pairs.append(Pair(
                q_hash=q_hash,
                traj_id_win=win_exp.traj_id,
                traj_id_lose=lose_exp.traj_id,
                exp_win=win_exp,
                exp_lose=lose_exp,
            ))

        return pairs

    def sample_individual(self, n: int) -> List[Experience]:
        """Return up to *n* individual Experience objects (uniform random, no pairing).

        Used for JOA computation — we only need individual samples, not pairs.
        Excludes y=-1 (format error) experiences.
        """
        all_exps = [
            exp
            for trajs in self._data.values()
            for exp in trajs.values()
            if exp.outcome >= 0
        ]
        if len(all_exps) <= n:
            return all_exps
        return random.sample(all_exps, n)

    def extend(self, experiences: List[Experience]) -> None:
        for exp in experiences:
            self.add(exp)

    def __len__(self) -> int:
        return self._total

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ucb_score(self, exp: Experience, T: int) -> float:
        # M1: max(T, math.e) ensures log is always positive
        return (
            self.lambda_err * abs(exp.outcome - exp.v_pred)
            + self.lambda_exp * math.sqrt(math.log(max(T, math.e)) / (exp.n_sampled + 1))
        )

    def _ucb_sample(self, candidates: List[Experience], T: int) -> Experience:
        """Weighted-random sample by UCB score (higher score = higher probability)."""
        scores = [max(self._ucb_score(e, T), 1e-8) for e in candidates]
        total  = sum(scores)
        probs  = [s / total for s in scores]
        return random.choices(candidates, weights=probs, k=1)[0]

    def _evict_worst_in_q(self, q_hash: str) -> None:
        """Remove the lowest-UCB trajectory from q_hash."""
        T = max(self.global_sample_steps, 1)
        trajs = self._data[q_hash]
        worst_id = min(trajs, key=lambda tid: self._ucb_score(trajs[tid], T))
        del trajs[worst_id]
        self._total -= 1

    def _evict_oldest_global(self) -> None:
        """M5: Remove the globally oldest trajectory (by timestamp), not random.

        This prevents the buffer from accumulating stale experiences from early
        training when the Actor was weak.  Using random eviction would keep
        old low-quality experiences alive indefinitely.
        """
        oldest_q: Optional[str] = None
        oldest_tid: Optional[str] = None
        oldest_ts: int = int(1e18)

        for q_hash, trajs in self._data.items():
            for tid, exp in trajs.items():
                if exp.timestamp < oldest_ts:
                    oldest_ts = exp.timestamp
                    oldest_q   = q_hash
                    oldest_tid = tid

        if oldest_q is not None and oldest_tid is not None:
            del self._data[oldest_q][oldest_tid]
            self._total -= 1
            if not self._data[oldest_q]:
                del self._data[oldest_q]

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @staticmethod
    def make_experience(
        context_text: str,
        prompt_text: str,
        question: str,
        strategy: str,
        outcome: int,
        timestamp: int = 0,
    ) -> Experience:
        """Create a new Experience with a fresh UUID traj_id."""
        return Experience(
            traj_id=str(uuid.uuid4()),
            context_text=context_text,
            prompt_text=prompt_text,
            question=question,
            strategy=strategy,
            outcome=outcome,
            timestamp=timestamp,
        )
