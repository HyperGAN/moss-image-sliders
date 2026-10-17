"""Portable adapter-only safetensors; no pickle or base weights."""

import json
from pathlib import Path

from safetensors.torch import load_file, save_file

from . import MODEL_ID, MODEL_REVISION
from .lora import Slider


def save_slider(slider, directory, training=None):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=False)
    manifest = {"format": "moss-image-sliders-v1", "base_model": MODEL_ID,
                "revision": MODEL_REVISION, **slider.manifest(), "training": training or {}}
    save_file({k: v.detach().float().cpu().contiguous() for k, v in slider.state_dict().items()},
              str(path / "adapter.safetensors"), metadata={"format": manifest["format"]})
    (path / "slider.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_slider(transformer, directory):
    path = Path(directory)
    manifest = json.loads((path / "slider.json").read_text())
    if (manifest["format"], manifest["base_model"], manifest["revision"]) != (
            "moss-image-sliders-v1", MODEL_ID, MODEL_REVISION):
        raise ValueError("Adapter format, model or pinned revision mismatch")
    slider = Slider(transformer, manifest["rank"], manifest["alpha"], manifest["targets"])
    try:
        state = load_file(str(path / "adapter.safetensors"))
        if any(not tensor.isfinite().all() for tensor in state.values()):
            raise ValueError("Adapter contains non-finite weights")
        slider.load_state_dict(state, strict=True)
    except Exception:
        slider.detach()
        raise
    return slider
