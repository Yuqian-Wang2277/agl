"""
Recover missing WandB metrics from output.log.

Parses lines like:
  step:216 - training/reward:0.789 - actor/lr:1e-06 - ...
and uploads them to the existing WandB run.

Usage:
  # Upload steps >= 188 that are missing from WandB:
  python recover_wandb.py --log-file wandb/run-20260406_202104-6m77ne9f/files/output.log \
                          --run-id 6m77ne9f \
                          --project StrategyGeneration \
                          --entity wangyuqian202405-personal \
                          --from-step 188

  # After training finishes, run again to catch all remaining steps.
"""

import argparse
import re
import sys


def parse_step_line(line: str) -> tuple[int, dict] | None:
    """Parse a metrics line from output.log. Returns (step, metrics) or None."""
    # Match lines that start with "step:NNN - key:val - key:val ..."
    if not line.startswith("step:"):
        return None
    parts = line.strip().split(" - ")
    if not parts:
        return None
    metrics = {}
    step = None
    for part in parts:
        part = part.strip()
        if ":" not in part:
            continue
        k, _, v = part.partition(":")
        k = k.strip()
        v = v.strip()
        if k == "step":
            try:
                step = int(v)
            except ValueError:
                return None
        else:
            try:
                metrics[k] = float(v)
            except ValueError:
                metrics[k] = v
    if step is None:
        return None
    return step, metrics


def main():
    parser = argparse.ArgumentParser(description="Recover WandB metrics from output.log")
    parser.add_argument("--log-file", required=True, help="Path to output.log")
    parser.add_argument("--run-id", required=True, help="WandB run ID (e.g. 6m77ne9f)")
    parser.add_argument("--project", required=True, help="WandB project name")
    parser.add_argument("--entity", required=True, help="WandB entity/username")
    parser.add_argument("--from-step", type=int, default=188,
                        help="Only upload steps >= this value (default: 188)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and print steps without uploading")
    args = parser.parse_args()

    # Collect all steps from log
    all_steps: dict[int, dict] = {}
    with open(args.log_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            result = parse_step_line(line)
            if result is not None:
                step, metrics = result
                if step >= args.from_step:
                    all_steps[step] = metrics

    if not all_steps:
        print(f"No steps >= {args.from_step} found in {args.log_file}")
        sys.exit(0)

    sorted_steps = sorted(all_steps.keys())
    print(f"Found {len(sorted_steps)} steps to upload: {sorted_steps[0]} ~ {sorted_steps[-1]}")

    if args.dry_run:
        for s in sorted_steps:
            print(f"  step={s}: {list(all_steps[s].keys())[:5]}...")
        print("Dry run complete, no data uploaded.")
        return

    import wandb

    run = wandb.init(
        id=args.run_id,
        project=args.project,
        entity=args.entity,
        resume="allow",
    )

    print(f"Uploading {len(sorted_steps)} steps to run {args.run_id}...")
    for step in sorted_steps:
        metrics = all_steps[step]
        # Use commit=True and explicit step
        wandb.log(metrics, step=step, commit=True)
        print(f"  Uploaded step {step} ({len(metrics)} metrics)")

    wandb.finish()
    print("Done. Check WandB for updated charts.")


if __name__ == "__main__":
    main()
