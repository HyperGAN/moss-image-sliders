"""MOSS bindings to the installed shared ParticleGAN 0.9 game.

Model loading and GPU admission belong to the caller. This module implements no
adversarial loss, optimizer, sampler, or checkpoint replay of its own.
"""

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import uuid

import torch

from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.pg09_backends import (
    MossResponseAdapter, ResponseRow, checkpoint_with_adapter_scale,
)
from particle_sliders.pg09_moss_campaign import CampaignFieldCritic

from . import MODEL_ID, MODEL_REVISION
from .lora import Slider
from .training import sample_training_state

OBJECTIVE = "pg09_gan_only"


@dataclass
class GanRun:
    backend: object
    game: ConditionalPG09

    @property
    def slider(self):
        return self.backend.slider

    def step(self):
        return self.game.step()


def build_run(pipe, contexts, config, *, checkpointing=True, model_identity=None):
    """Validate before attachment and roll back hooks/flags after a failed build."""
    for key, lower, upper in (("rank", 1, 128), ("batch_size", 2, 512),
                              ("rollout_steps", 2, 50), ("steps", 1, 10000),
                              ("seed", 0, 2**32 - 1), ("resolution", 16, 2048)):
        if type(config.get(key)) is not int or not lower <= config[key] <= upper:
            raise ValueError("Invalid GAN recipe field: " + key)
    if config["resolution"] % 16 or any(
            not isinstance(config.get(key), (int, float)) or
            not math.isfinite(config[key]) or config[key] <= 0 for key in ("lr", "strength")):
        raise ValueError("Invalid GAN resolution, learning rate or strength")
    blocks = list(pipe.transformer.layers)
    if any(getattr(block, "_pg09_scale_checkpoint", False) for block in blocks):
        raise ValueError("Build a new GAN run with an unwrapped transformer")
    flags = [(p, p.requires_grad) for p in pipe.transformer.parameters()]
    forwards = [(block, block.forward) for block in blocks]
    owner = []
    try:
        return _build_run(pipe, contexts, config, checkpointing=checkpointing,
                          model_identity=model_identity, owner=owner)
    except Exception:
        for slider in owner:
            slider.detach()
        for block, forward in forwards:
            block.forward = forward
            if hasattr(block, "_pg09_scale_checkpoint"):
                delattr(block, "_pg09_scale_checkpoint")
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        raise


def _build_run(pipe, contexts, config, *, checkpointing, model_identity, owner):
    """Fixed frozen neutral rollouts, paired signs, and useful context/time features.

    Each scheduler position receives a separately seeded frozen neutral rollout.
    Both signs use the identical latent/time and positive-minus-negative teacher.
    The shared game owns independent balanced D/G row and reference-noise streams.
    """
    if config.get("objective") != OBJECTIVE or config.get("pg_version") != "0.9.1":
        raise ValueError("An explicit PG 0.9 GAN-only recipe is required")
    device = pipe.transformer.device
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config["seed"])
        slider = Slider(pipe.transformer, rank=config["rank"])
        owner.append(slider)
        identity = dict(model_identity or {"model": MODEL_ID, "revision": MODEL_REVISION})
        if hasattr(pipe, "_moss_training_context_identity"):
            identity["conditioning"] = pipe._moss_training_context_identity
        backend = MossResponseAdapter(
            pipe.transformer, slider, model_identity=identity,
        )
        generator = torch.Generator("cpu").manual_seed(config["seed"])
        rows = []
        neutral, positive, negative = contexts
        pooled = (neutral[0].float() * neutral[1][..., None]).sum(1)
        pooled = pooled / neutral[1].sum(1, keepdim=True)
        pooled = pooled / pooled.square().mean().sqrt().clamp_min(1e-12)
        conditions = []
        for position in range(config["rollout_steps"]):
            latent, time = sample_training_state(
                pipe, slider, neutral, generator, config["resolution"],
                config["rollout_steps"], position,
            )
            row = ResponseRow(neutral, positive, negative, latent, timestep=time,
                              strength=config["strength"], guidance=1.)
            for sign in (1., -1.):
                rows.append(backend.prepare(replace(row, scale=sign)))
                suffix = pooled.new_tensor([[float(time) / 1000., sign,
                                             config["strength"], 1. / 3., 0.]])
                conditions.append(torch.cat((pooled, suffix), dim=1))
        conditions = torch.cat(conditions).detach()
        scale = torch.cat([(r.target - r.neutral).flatten() for r in rows])
        scale = float(scale.double().square().mean().sqrt())
        if not 0 < scale < float("inf"):
            slider.detach()
            raise ValueError("Frozen training edit has no finite nonzero response scale")
        critic = CampaignFieldCritic(rows[0].target.shape[1], conditions.shape[1],
                                    config.get("critic_width", 32)).to(device)
    if checkpointing:
        for block in pipe.transformer.layers:
            checkpoint_with_adapter_scale(block, backend.network)
    game = ConditionalPG09(
        backend, rows, critic, conditions=conditions, particles=None,
        response_scale=scale, response_domain="continuous_field",
        preset="bispf_optimized", sampling="balanced",
        batch_size=config["batch_size"], total_steps=config["steps"], seed=config["seed"],
        recipe_overrides={"lr": config["lr"]},
    )
    return GanRun(backend, game)


def prepare_cached_rows(cache):
    """Validate cache identity and shared runtime before loading a real model."""
    from particle_sliders.pg09_moss_campaign import MossCampaignRows
    from particle_sliders.pg09 import validate_pg09_domain
    provider = MossCampaignRows(cache)
    if provider.recipe["model_revision"] != MODEL_REVISION or len(provider) < 2:
        raise ValueError("Campaign cache requires the pinned MOSS revision and nonempty signed rows")
    validate_pg09_domain(provider.response_shape, response_domain="continuous_field")
    return provider


def build_cached_run(transformer, cache, *, total_steps=3840, batch_size=8,
                     model_identity=None, provider=None):
    """Reuse the shared read-only campaign provider and its exact conditioning."""
    if type(batch_size) is not int or not 2 <= batch_size <= 512:
        raise ValueError("batch_size must be an integer in [2,512]")
    if type(total_steps) is not int or not 1 <= total_steps <= 3840:
        raise ValueError("Campaign total_steps must be an integer in [1,3840]")
    from particle_sliders.pg09_moss_campaign import build_campaign_run
    provider = prepare_cached_rows(cache) if provider is None else provider
    run = build_campaign_run(
        transformer, provider, response_scale=provider.response_scale(),
        total_steps=total_steps, batch_size=batch_size,
        model_identity=model_identity or {"model": MODEL_ID, "revision": MODEL_REVISION},
    )
    return GanRun(run.backend, run.game)


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return value


def save_checkpoint(run, root):
    """Immutable full shared-game packet, distinct from inference-only adapters."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    step = run.game.completed_steps
    target = root / f"step-{step:06d}"
    if target.exists():
        raise FileExistsError(target)
    pending = root / (".pending-" + uuid.uuid4().hex)
    pending.mkdir()
    try:
        packet = _cpu(run.game.state_dict())
        torch.save(packet, pending / "training.pt")
        metadata = dict(schema="moss-pg09-checkpoint-v1", objective=OBJECTIVE,
                        step=step, sha256=hashlib.sha256(
                            (pending / "training.pt").read_bytes()).hexdigest())
        (pending / "checkpoint.json").write_text(json.dumps(metadata, indent=2) + "\n")
        pending.rename(target)
    finally:
        if pending.exists():
            shutil.rmtree(pending)
    return target


def restore_checkpoint(run, directory):
    """Read RNG tensors on CPU; shared core validates before mutating state."""
    directory = Path(directory)
    meta = json.loads((directory / "checkpoint.json").read_text())
    if (meta.get("schema") != "moss-pg09-checkpoint-v1" or
            meta.get("objective") != OBJECTIVE or type(meta.get("step")) is not int):
        raise ValueError("Only full PG09 GAN checkpoints can resume training")
    path = directory / "training.pt"
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("GAN checkpoint checksum mismatch")
    packet = torch.load(path, map_location="cpu", weights_only=True)
    if packet.get("completed_steps") != meta["step"]:
        raise ValueError("GAN checkpoint step differs")
    run.game.load_state_dict(packet)
    return meta["step"]


def train_to_budget(run, output, recipe, *, resume=None, checkpoint_every=10):
    """Caller opts into execution; no background job or model publication."""
    from .artifacts import save_gan_slider
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError("checkpoint_every must be a positive integer")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if resume is not None:
        restore_checkpoint(run, resume)
    output.mkdir(parents=True)
    provenance = dict(objective=OBJECTIVE, pg_version="0.9.1", preset="bispf_optimized",
                      recipe=recipe, resumed_from=str(resume) if resume else None,
                      initial_completed_steps=run.game.completed_steps)
    (output / "run.json").write_text(json.dumps(provenance, indent=2) + "\n")
    while run.game.completed_steps < run.game.recipe.total_steps:
        metric = run.step()
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metric) + "\n")
        print(json.dumps(metric), flush=True)
        if (run.game.completed_steps % checkpoint_every == 0 or
                run.game.completed_steps == run.game.recipe.total_steps):
            save_checkpoint(run, output / "checkpoints")
    save_gan_slider(run.slider, output / "adapter", provenance,
                    core_config=run.game.export_state_dict()["config"],
                    completed_steps=run.game.completed_steps)
