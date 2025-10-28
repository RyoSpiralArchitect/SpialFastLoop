# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ryō

__version__ = "0.1.1"
from .engine import FastTrainer, recommended_dataloader
from . import utils
from .metrics import GLOBAL_NORMALIZATION_METRICS, NormalizationMetricsCollector

__all__ = [
    "FastTrainer",
    "GLOBAL_NORMALIZATION_METRICS",
    "NormalizationMetricsCollector",
    "recommended_dataloader",
    "utils",
    "__version__",
]
