"""Portable adapter-only safetensors; no pickle or base weights."""

import json
import hashlib
import math
from pathlib import Path
import shutil
import tempfile

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from . import MODEL_ID, MODEL_REVISION
from .lora import Slider, TARGET

V1 = "moss-image-sliders-v1"
V2 = "moss-image-sliders-v2"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_gan_config(config, names, shapes, completed_steps):
    """Require the shared PG09 adapter-GAN signature, not a legacy MSE recipe."""
    from particle_sliders.pg09 import PG_VERSION, PG_WHEEL_SHA256
    expected = dict(pg_version=PG_VERSION, pg_release_wheel_sha256=PG_WHEEL_SHA256,
                    generator_arm="adapter_gan", response_domain="continuous_field",
                    objective="g_adv+public_prior;d_adv+public_penalty")
    recipe = config.get("recipe", {})
    backend = config.get("backend", {})
    if (any(config.get(k) != value for k, value in expected.items())
            or recipe.get("name") != "bispf_optimized" or recipe.get("model") != "gan"
            or backend.get("family") != "moss"
            or backend.get("cfg_adapter_branch") != "conditional_only"
            or backend.get("negative_layer_skip") is not False
            or backend.get("adapter_names") != list(names)):
        raise ValueError("Expected the shared ParticleGAN 0.9 adapter-GAN config and MOSS targets")
    expected_schema = {f"weights.{i}": [list(shape), "torch.float32"]
                       for i, shape in enumerate(shapes)}
    if json.loads(json.dumps(config.get("network_schema"))) != expected_schema:
        raise ValueError("GAN config adapter network schema differs")
    if (type(completed_steps) is not int or type(recipe.get("total_steps")) is not int
            or not 0 <= completed_steps <= recipe["total_steps"]):
        raise ValueError("GAN completed_steps is outside the shared training budget")
    for key in ("pg_sources_sha256", "rows_sha256", "conditions_sha256"):
        value = config.get(key)
        if value != "bound-by-provider-identity" and (
                not isinstance(value, str) or len(value) != 64):
            raise ValueError("GAN config lacks shared source/conditioning identity: " + key)
    if not config.get("implementation_sources"):
        raise ValueError("GAN config lacks shared implementation source identity")


def save_gan_slider(slider, directory, training, *, core_config, completed_steps):
    """Portable GAN adapter with its actual shared-core protocol provenance."""
    named = list(slider.named_parameters())
    _validate_gan_config(core_config, [name for name, _ in named],
                         [tuple(value.shape) for _, value in named], completed_steps)
    state = slider.state_dict()
    if any(value.dtype != torch.float32 or not value.isfinite().all() for value in state.values()):
        raise ValueError("GAN adapter leaves must be finite float32 tensors")
    path = Path(directory)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    manifest = dict(format=V2, base_model=MODEL_ID, revision=MODEL_REVISION,
                    **slider.manifest(), objective="pg09_gan_only", pg_version="0.9.1",
                    preset="bispf_optimized", core_config=core_config,
                    completed_steps=completed_steps, training=training or {},
                    adapter_sha256="0" * 64)
    # Validate every provenance field and freeze its JSON snapshot before creating
    # any output. The actual file checksum is filled in after staging the weights.
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=f".{path.name}.pending-", dir=path.parent))
    reserved = False
    try:
        weights = pending / "adapter.safetensors"
        save_file({k: v.detach().cpu().contiguous() for k, v in state.items()},
                  str(weights), metadata={"format": V2})
        manifest["adapter_sha256"] = _sha256(weights)
        (pending / "slider.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        # Exclusively reserve the destination, then publish the manifest last as
        # the commit marker. Readers reject incomplete directories without it.
        path.mkdir(exist_ok=False)
        reserved = True
        weights.replace(path / weights.name)
        (pending / "slider.json").replace(path / "slider.json")
    except Exception:
        if reserved:
            shutil.rmtree(path)
        raise
    finally:
        if pending.exists():
            shutil.rmtree(pending)


def save_slider(slider, directory, training=None):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=False)
    manifest = {"format": V1, "base_model": MODEL_ID,
                "revision": MODEL_REVISION, **slider.manifest(), "training": training or {}}
    save_file({k: v.detach().float().cpu().contiguous() for k, v in slider.state_dict().items()},
              str(path / "adapter.safetensors"), metadata={"format": manifest["format"]})
    (path / "slider.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_slider(transformer, directory):
    path = Path(directory)
    if path.is_file() or not (path / "slider.json").is_file():
        raise ValueError("Expected a portable MOSS v1/v2 adapter directory. Shared worker exports "
                         "must be validated with MossCampaignRun.load_adapter and their original "
                         "cache/model identity, then exported as a portable v2 adapter.")
    manifest = json.loads((path / "slider.json").read_text())
    if (manifest["format"] not in (V1, V2)
            or (manifest["base_model"], manifest["revision"]) != (MODEL_ID, MODEL_REVISION)):
        raise ValueError("Adapter format, model or pinned revision mismatch")
    rank, alpha = manifest["rank"], manifest["alpha"]
    if type(rank) is not int or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("rank and alpha must be positive and finite")
    modules = dict(transformer.named_modules())
    names = sorted(name for name, module in modules.items()
                   if TARGET.fullmatch(name) and isinstance(module, nn.Linear))
    if not names or manifest["targets"] != names:
        raise ValueError("Adapter targets differ from the frozen MOSS transformer")
    schema = {}
    for i, name in enumerate(names):
        schema[f"residuals.{i}.down.weight"] = (rank, modules[name].in_features)
        schema[f"residuals.{i}.up.weight"] = (modules[name].out_features, rank)
    if manifest["format"] == V2:
        if ((manifest.get("objective"), manifest.get("pg_version"), manifest.get("preset"))
                != ("pg09_gan_only", "0.9.1", "bispf_optimized")):
            raise ValueError("GAN adapter objective, ParticleGAN version or preset mismatch")
        if manifest.get("adapter_sha256") != _sha256(path / "adapter.safetensors"):
            raise ValueError("GAN adapter checksum mismatch")
        _validate_gan_config(manifest.get("core_config", {}), schema.keys(), schema.values(),
                             manifest.get("completed_steps"))
    state = load_file(str(path / "adapter.safetensors"))
    if state.keys() != schema.keys() or any(tuple(t.shape) != schema[k] for k, t in state.items()):
        raise RuntimeError("Adapter tensor keys or shapes differ from the target schema")
    if any(t.dtype != torch.float32 or not t.isfinite().all() for t in state.values()):
        raise ValueError("Adapter contains non-finite or non-float32 weights")
    # The full artifact is validated before hooks are attached or base flags change.
    slider = Slider(transformer, rank, alpha, manifest["targets"])
    try:
        slider.load_state_dict(state, strict=True)
    except Exception:
        slider.detach()
        raise
    return slider
