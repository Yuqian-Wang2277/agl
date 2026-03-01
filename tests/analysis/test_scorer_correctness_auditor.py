"""Tests for analysis/scorer_correctness_auditor.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "analysis" / "scorer_correctness_auditor.py"
    spec = importlib.util.spec_from_file_location("scorer_correctness_auditor", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_audit_prefers_global_files_and_writes_report(tmp_path: Path) -> None:
    module = _load_module()
    val_dir = tmp_path / "validation_outputs"
    val_dir.mkdir()

    global_payload = [
        {
            "problem_type": "alpha",
            "output": {"strategy_extracted": "use elimination"},
            "reward": {"scorer": 0.2, "correctness": 0.0, "final": 0.18},
        },
        {
            "problem_type": "alpha",
            "output": {"strategy_extracted": "compare facts"},
            "reward": {"scorer": 0.9, "correctness": 1.0, "final": 0.91},
        },
    ]
    shard_payload = [
        {
            "problem_type": "alpha",
            "output": {"strategy_extracted": "this shard should be ignored when global exists"},
            "reward": {"scorer": 0.5, "correctness": 0.0, "final": 0.45},
        }
    ]
    (val_dir / "validation_global_step0.json").write_text(
        json.dumps(global_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    (val_dir / "validation_step0_worker1.json").write_text(
        json.dumps(shard_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    output_dir = tmp_path / "analysis"
    report = module.run_audit(str(val_dir), str(output_dir))

    assert report["n_records"] == 2
    assert report["parse_rate"] == 1.0
    assert report["top_scorer_bin_size"] == 1
    assert report["top_scorer_bin_correctness"] == 1.0
    assert report["approx_grounded_proxy_distribution"] == {"0.00": 1, "1.00": 1}
    assert (output_dir / "audit_report.json").exists()


def test_counterfactual_audit_reports_help_harm_neutral() -> None:
    module = _load_module()
    records = [
        {
            "input": {"problem": "p1", "ground_truth": "A"},
            "output": {"strategy_extracted": "help"},
        },
        {
            "input": {"problem": "p2", "ground_truth": "B"},
            "output": {"strategy_extracted": "harm"},
        },
        {
            "input": {"problem": "p3", "ground_truth": "C"},
            "output": {"strategy_extracted": "neutral"},
        },
    ]

    def fake_answer_model(strategy: str, problem: str) -> str:
        if problem == "p1":
            return "A" if strategy else "wrong"
        if problem == "p2":
            return "wrong" if strategy else "B"
        return "C"

    report = module.counterfactual_audit(records, fake_answer_model, n_sample=3, seed=0)

    assert report["n_records"] == 3
    assert report["help_rate"] == 0.3333
    assert report["harm_rate"] == 0.3333
    assert report["neutral_rate"] == 0.3333
