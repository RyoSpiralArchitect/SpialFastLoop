# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Ryō

__version__ = "0.1.1"
from .engine import FastTrainer, recommended_dataloader
from . import utils
__all__ = ["FastTrainer", "recommended_dataloader", "utils", "__version__"]
