"""Cached signed velocity distillation, balanced sampling, and local resumable state.

The loss is the original MOSS slider MSE. No adversarial/particle objective is used.
Training checkpoints are private local optimizer files, not distributable adapters.
"""

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import uuid

import torch
from torch.nn import functional as F

from .artifacts import save_slider
from .training import velocity


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class StateSampler:
    """Shuffle the full Cartesian dataset; row and time cannot become modulo-coupled."""

    def __init__(self, size, seed):
        if size < 1:
            raise ValueError("Empty training state set")
        self.size = size
        self.rng = random.Random(seed)
        self.order = []
        self.cursor = 0
        self.epochs = 0

    def take(self, count):
        result = []
        for _ in range(count):
            if self.cursor == len(self.order):
                self.order = list(range(self.size))
                self.rng.shuffle(self.order)
                self.cursor = 0
                self.epochs += 1
            result.append(self.order[self.cursor])
            self.cursor += 1
        return result

    def state_dict(self):
        return dict(
            size=self.size,
            rng=self.rng.getstate(),
            order=self.order,
            cursor=self.cursor,
            epochs=self.epochs,
        )

    def load_state_dict(self, state):
        if state["size"] != self.size:
            raise ValueError("Training dataset changed on resume")
        self.rng.setstate(state["rng"])
        self.order, self.cursor, self.epochs = state["order"], state["cursor"], state["epochs"]


def cached_backward(model, slider, sample, context, divisor=1):
    """Accumulate one independent state; caller steps Adam once per effective batch."""
    losses = []
    for sign in (1, -1):
        with slider.at(sign):
            prediction = velocity(model, sample["latent"], sample["time"], context)
            target = sample["base"] + sign * sample["delta"]
            loss = F.mse_loss(prediction, target) / 2
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite signed training loss")
            (loss / divisor).backward()
            losses.append(float(loss.detach()))
    return sum(losses)


@torch.no_grad()
def target_bundle(pipe, slider, neutral, positive, seed, resolution=512, steps=10):
    """Both frozen paths from identical noise, both prompt velocities at every position."""
    device = pipe.transformer.device
    noise = torch.randn(
        (1, pipe.transformer.config.latent_channels, resolution // 16, resolution // 16),
        generator=torch.Generator("cpu").manual_seed(seed),
    ).to(device)
    paths = {key: [] for key in ("latent", "base", "delta")}
    with slider.at(0):
        for positive_path in (False, True):
            pipe.scheduler.set_timesteps(steps, device=device)
            latent = noise.clone()
            values = {key: [] for key in paths}
            for time in pipe.scheduler.timesteps:
                base = velocity(pipe.transformer, latent, time, neutral)
                edited = velocity(pipe.transformer, latent, time, positive)
                for key, value in dict(latent=latent, base=base, delta=edited - base).items():
                    if not value.isfinite().all():
                        raise RuntimeError("Non-finite frozen target")
                    values[key].append(value.detach().cpu())
                latent = pipe.scheduler.step(
                    edited if positive_path else base, time, latent, return_dict=False
                )[0]
            for key in paths:
                paths[key].append(torch.stack(values[key]))
    return {
        **{key: torch.stack(value) for key, value in paths.items()},
        "time": pipe.scheduler.timesteps.detach().cpu(),
    }


@torch.no_grad()
def edit_diagnostics(model, slider, fixtures):
    """Fixed held-out states, separately reporting both signs; no fitted normalization."""
    fixtures = list(fixtures)
    if not fixtures:
        raise ValueError("No fixed diagnostic states")
    result = {}
    for sign in (1, -1):
        error = teacher_energy = student_energy = dot = count = 0.0
        for sample, context in fixtures:
            with slider.at(sign):
                student = (
                    velocity(model, sample["latent"], sample["time"], context) - sample["base"]
                )
            teacher = sign * sample["delta"]
            error += float((student - teacher).square().sum())
            teacher_energy += float(teacher.square().sum())
            student_energy += float(student.square().sum())
            dot += float((student * teacher).sum())
            count += teacher.numel()
        result[str(sign)] = dict(
            relative_edit_mse=error / max(teacher_energy, 1e-30),
            teacher_rms=(teacher_energy / count) ** 0.5,
            student_rms=(student_energy / count) ** 0.5,
            edit_rms_ratio=(student_energy / max(teacher_energy, 1e-30)) ** 0.5,
            edit_cosine=dot / max((student_energy * teacher_energy) ** 0.5, 1e-30),
        )
    return result


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


def save_training_checkpoint(root, step, slider, optimizer, sampler, identity, metrics=None):
    """Publish a complete immutable directory before advancing LATEST; never overwrite."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = f"step-{step:06d}"
    destination = root / name
    if destination.exists():
        raise FileExistsError(destination)
    temporary = root / (".pending-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        save_slider(slider, temporary / "adapter", {"step": step, "identity": identity})
        state = dict(
            format=1,
            step=step,
            identity=identity,
            slider=_cpu(slider.state_dict()),
            optimizer=_cpu(optimizer.state_dict()),
            sampler=sampler.state_dict(),
            cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            metrics=metrics or {},
        )
        with (temporary / "training.pt").open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        checksum = hashlib.sha256((temporary / "training.pt").read_bytes()).hexdigest()
        atomic_json(
            temporary / "checkpoint.json",
            dict(step=step, identity=identity, training_sha256=checksum, metrics=metrics or {}),
        )
        temporary.rename(destination)
        atomic_json(root / "LATEST.json", {"directory": name, "step": step})
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def latest_training_checkpoint(root, identity):
    """Recover a complete directory even if interruption preceded the LATEST update."""
    candidates = sorted(Path(root).glob("step-*"), reverse=True)
    for directory in candidates:
        metadata_path = directory / "checkpoint.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        if metadata["identity"] != identity:
            raise ValueError("Run identity changed on resume")
        path = directory / "training.pt"
        if (
            path.exists()
            and hashlib.sha256(path.read_bytes()).hexdigest() == metadata["training_sha256"]
        ):
            return directory
    return None


def restore_training_checkpoint(directory, slider, optimizer, sampler, identity):
    directory = Path(directory)
    metadata = json.loads((directory / "checkpoint.json").read_text())
    if metadata["identity"] != identity:
        raise ValueError("Run identity changed on resume")
    path = directory / "training.pt"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["training_sha256"]:
        raise ValueError("Training checkpoint checksum mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["format"] != 1 or state["identity"] != identity or state["step"] != metadata["step"]:
        raise ValueError("Training checkpoint identity mismatch")
    if any(not item.isfinite().all() for item in state["slider"].values()):
        raise ValueError("Non-finite adapter in training checkpoint")
    slider.load_state_dict(state["slider"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    sampler.load_state_dict(state["sampler"])
    torch.set_rng_state(state["cpu_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["step"]
