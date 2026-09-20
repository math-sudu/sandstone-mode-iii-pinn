"""Config loading + global seeding (deterministic, config-driven)."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def project_root() -> Path:
    """Absolute project root (.../baxi-cohesive), independent of cwd."""
    # src/baxi_pinn/config.py -> parents[2] == project root
    return Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {p} did not parse to a mapping")
    return cfg


def set_seed(seed: int) -> None:
    """Make the run reproducible to the float tolerances the acceptance asks for."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # CPU-only smoke does not need the cudnn flags, but set them for the cuda path.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def resolve_device(cfg: Dict[str, Any]) -> torch.device:
    want = str(cfg.get("device", "cpu")).lower()
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class OutPaths:
    """Resolved (absolute) output paths for a run, derived from cfg['output']."""

    root: Path
    synthetic_twin_recovery: Path
    metrics: Path
    identified_law: Path
    bootstrap_samples: Path

    @staticmethod
    def from_cfg(cfg: Dict[str, Any]) -> "OutPaths":
        root = project_root()
        out = cfg.get("output", {})

        def _abs(rel: str) -> Path:
            return root / rel

        return OutPaths(
            root=root,
            synthetic_twin_recovery=_abs(
                out.get("synthetic_twin_recovery", "results/runs/synthetic_twin/recovery.json")
            ),
            metrics=_abs(out.get("metrics", "results/runs/smoke/metrics.json")),
            identified_law=_abs(out.get("identified_law", "results/runs/smoke/identified_law.json")),
            bootstrap_samples=_abs(
                out.get("bootstrap_samples", "results/runs/smoke/bootstrap_samples.json")
            ),
        )
