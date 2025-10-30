"""Utilities to collect recurring quality metrics for SpiralFastLoop."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

from spiralfastloop import FastTrainer
from spiralfastloop.utils import dataloader_from_dataset


@dataclass
class CommandResult:
    """Capture stdout/stderr and runtime for a subprocess command."""

    command: Sequence[str]
    returncode: int
    duration_s: float
    stdout: str
    stderr: str

    def ensure_ok(self) -> "CommandResult":
        if self.returncode != 0:
            command_str = " ".join(self.command)
            message = (
                f"Command {command_str} failed with code {self.returncode}:\n"
                f"{self.stderr}"
            )
            raise RuntimeError(message)
        return self


def _run(cmd: Sequence[str], *, env: dict[str, str] | None = None) -> CommandResult:
    start = time.perf_counter()
    proc = subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        env=env,
    )
    return CommandResult(
        command=list(cmd),
        returncode=proc.returncode,
        duration_s=time.perf_counter() - start,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def run_pytest_with_coverage(pytest_args: Iterable[str]) -> tuple[CommandResult, float]:
    coverage_json = Path("coverage.json")
    if coverage_json.exists():
        coverage_json.unlink()
    cmd = ["pytest", "--cov", "--cov-report", "json:coverage.json", *pytest_args]
    result = _run(cmd)
    if result.returncode != 0:
        return result, 0.0
    coverage = 0.0
    if coverage_json.exists():
        with coverage_json.open("r", encoding="utf-8") as fh:
            report = json.load(fh)
        try:
            totals = report["totals"]
            coverage = float(totals["percent_covered_display"])
        except (KeyError, ValueError, TypeError):
            coverage = 0.0
        finally:
            coverage_json.unlink()
    return result, coverage


def run_smoke_bench(*, steps: int = 8, feature_dim: int = 64) -> dict[str, float]:
    torch.manual_seed(0)
    samples = 1024
    inputs = torch.randn(samples, feature_dim)
    targets = torch.randint(0, 4, (samples,))
    loader = dataloader_from_dataset(
        torch.utils.data.TensorDataset(inputs, targets),
        batch_size=64,
        device="cpu",
        num_workers=0,
        persistent=False,
        shuffle=False,
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(feature_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 4),
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    trainer = FastTrainer(
        model,
        optimiser,
        grad_accum=1,
        log_interval=steps + 1,
        device="cpu",
        compile_mode="reduce-overhead",
    )
    metrics = trainer.train_one_epoch(loader, criterion, steps=steps)
    return {
        "samples_per_sec": float(metrics.get("samples_per_sec", 0.0)),
        "avg_loss": float(metrics.get("avg_loss", 0.0)),
        "steps": float(metrics.get("steps", 0)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pytest-args",
        nargs="*",
        default=(),
        help="Extra arguments forwarded to pytest.",
    )
    parser.add_argument(
        "--skip-bench",
        action="store_true",
        help="Do not run the lightweight throughput smoke test.",
    )
    args = parser.parse_args(argv)

    pytest_result, coverage_pct = run_pytest_with_coverage(args.pytest_args)
    summary: dict[str, object] = {
        "pytest": {
            "command": list(pytest_result.command),
            "returncode": pytest_result.returncode,
            "duration_s": pytest_result.duration_s,
        },
        "coverage_percent": coverage_pct,
    }

    if pytest_result.returncode != 0:
        print(pytest_result.stdout, file=sys.stdout)
        print(pytest_result.stderr, file=sys.stderr)
        print(json.dumps(summary, indent=2), file=sys.stdout)
        return pytest_result.returncode

    if not args.skip_bench:
        bench_metrics = run_smoke_bench()
        summary["bench"] = bench_metrics

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
