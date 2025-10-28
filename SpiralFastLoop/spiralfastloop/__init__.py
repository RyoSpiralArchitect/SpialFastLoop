# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ryō

__version__ = "0.1.1"
from . import utils
from .engine import FastTrainer, recommended_dataloader

__all__ = ["FastTrainer", "recommended_dataloader", "utils", "__version__"]
