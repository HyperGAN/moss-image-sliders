"""Signed residual LoRA adapted from particle-sliders' LoRAModule.

Original copyright (c) 2023 Rohit Gandikota, MIT; see LICENSES/particle-sliders-MIT.txt.
Forward hooks retain Linear.weight because the official MOSS forward reads it directly.
"""

from contextlib import contextmanager
import math
import re

from torch import nn

TARGET = re.compile(r"layers\.\d+\.proj\.(q|k|v|o)$")


class LowRankResidual(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.down = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device)
        self.up = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device)
        nn.init.kaiming_uniform_(self.down.weight, a=1)
        nn.init.zeros_(self.up.weight)
        self.factor = alpha / rank

    def forward(self, x):
        x = x.to(device=self.down.weight.device, dtype=self.down.weight.dtype)
        return self.up(self.down(x)) * self.factor


class Slider(nn.Module):
    """One adapter with reversible signed scale and strict MOSS attention targets."""

    def __init__(self, transformer, rank=4, alpha=None, targets=None):
        super().__init__()
        alpha = rank if alpha is None else alpha
        if type(rank) is not int or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("rank and alpha must be positive and finite")
        modules = dict(transformer.named_modules())
        names = sorted(name for name, module in modules.items()
                       if TARGET.fullmatch(name) and isinstance(module, nn.Linear))
        if not names or (targets is not None and list(targets) != names):
            raise ValueError("MOSS attention target set is empty or incompatible with checkpoint")
        if any(getattr(modules[name], "_moss_slider_attached", False) for name in names):
            raise ValueError("A slider is already attached to this transformer")
        transformer.requires_grad_(False)
        self.rank, self.alpha, self.targets = rank, float(alpha), names
        self.scale = 0.0
        self.residuals = nn.ModuleList([
            LowRankResidual(modules[name], rank, alpha) for name in names
        ])
        self._handles = []
        self._modules_attached = []
        for name, residual in zip(names, self.residuals):
            module = modules[name]
            module._moss_slider_attached = True
            self._modules_attached.append(module)
            self._handles.append(module.register_forward_hook(self._hook(residual)))

    def _hook(self, residual):
        def apply(module, args, output):
            if self.scale == 0:
                return output
            delta = residual(args[0]).to(device=output.device, dtype=output.dtype)
            return output + self.scale * delta
        return apply

    @contextmanager
    def at(self, value):
        value = float(value)
        if not math.isfinite(value) or abs(value) > 3:
            raise ValueError("Slider scale must be finite and between -3 and 3")
        previous, self.scale = self.scale, value
        try:
            yield self
        finally:
            self.scale = previous

    def detach(self):
        for handle in self._handles:
            handle.remove()
        for module in self._modules_attached:
            module._moss_slider_attached = False
        self._handles.clear()
        self._modules_attached.clear()

    def manifest(self):
        return {"rank": self.rank, "alpha": self.alpha, "targets": self.targets}
