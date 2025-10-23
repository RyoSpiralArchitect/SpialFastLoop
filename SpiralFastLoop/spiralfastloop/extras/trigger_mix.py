\
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, Tuple

import torch

@dataclass
class LossStdConfig:
    """Config for triggering hard-sample injection based on per-sample loss std."""
    std_threshold: float = 0.15
    inject_ratio: float = 0.08      # fraction of batch to add at most
    weight_alpha: float = 1.2       # weight for injected samples
    budget_frac: float = 0.03       # token/sample budget per epoch (approx)
    pulse_every: int = 800          # force a pulse every N steps
    max_injected_per_step: int = 128

class LossStdTrigger:
    """
    A generic trigger hook: when per-sample loss std is too low (learning too 'smooth'),
    call a provider to obtain extra 'hard' samples to inject into the current batch.
    Provider signature: provider(k:int, device:str, ctx:dict) -> (inputs, targets)
    """
    def __init__(self, provider: Callable[[int, str, Dict[str, Any]], Tuple[Any, Any]],
                 cfg: Optional[LossStdConfig] = None):
        self.provider = provider
        self.cfg = cfg or LossStdConfig()
        self.spent = 0  # approximate budget spent (samples)
        self.total = 0  # approximate total samples seen

    def __call__(self, ctx: Dict[str, Any]):
        loss_vec: torch.Tensor = ctx["loss_vec"].detach()
        device = ctx["device"]
        step = ctx.get("step", 0)

        B = loss_vec.numel()
        self.total += B

        coefvar = loss_vec.std() / (loss_vec.mean().abs() + 1e-8)
        need = (coefvar.item() <= self.cfg.std_threshold) or (step % self.cfg.pulse_every == 0)

        budget_ok = (self.spent <= self.cfg.budget_frac * max(1, self.total))
        if not (need and budget_ok):
            return None

        k = min(int(B * self.cfg.inject_ratio), self.cfg.max_injected_per_step)
        if k <= 0:
            return None

        extra_x, extra_y = self.provider(k, device, ctx)
        self.spent += k

        # weights: original ones at 1.0, injected at alpha
        w = torch.ones(B + k, device=loss_vec.device)
        w[-k:] = self.cfg.weight_alpha
        return type("TrigRes", (), {"extra_inputs": extra_x, "extra_targets": extra_y, "weights": w})
