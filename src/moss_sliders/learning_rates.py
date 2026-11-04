"""Explicit public PG rates for the adapter-only current MOSS cohort."""
import math


def recipe_overrides(config):
    if "prior_lr_mult" in config:
        raise ValueError("MOSS is adapter-only; prior_lr_mult requires a particle bank")
    result = {}
    for key in ("lr", "d_lr_mult"):
        if key not in config:
            if key == "lr":
                raise ValueError("MOSS requires an explicit lr in its training configuration")
            continue
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(key + " must be a finite positive number")
        result[key] = value
    if "d_lr_mult" in result and not math.isfinite(result["lr"] * result["d_lr_mult"]):
        raise ValueError("resolved critic learning rate must be finite")
    return result
