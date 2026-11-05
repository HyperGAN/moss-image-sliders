"""Installed unified dispatcher calls the product's actual shared GAN factory."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from moss_sliders.execution import validate_training


def fixture_recipe():
    return dict(neutral="neutral", positive="positive", negative="negative", dummy=True,
                objective="pg09_gan_only", pg_version="0.9.5", preset="bispf_optimized",
                steps=2, rank=2, lr=.001, resolution=32, rollout_steps=2, strength=1.,
                seed=17, max_sequence_length=16, batch_size=2)


def test_explicit_fixture_and_version_gates_precede_model_or_game_construction():
    recipe = fixture_recipe()
    del recipe["dummy"]
    with pytest.raises(ValueError, match="dummy=true"):
        validate_training(recipe, fixture=True)
    recipe.update(dummy=True, steps=21)
    with pytest.raises(ValueError, match="bounded"):
        validate_training(recipe, fixture=True)
    recipe.update(steps=2, pg_version="0.9.1")
    with pytest.raises(ValueError, match="GAN-only"):
        validate_training(recipe, fixture=True)


def test_installed_unified_profile_check_train_and_resume(tmp_path):
    from particle_sliders.execution.request import resolve_request
    from particle_sliders.execution.local import check_request, execute_resolved_request as execute
    config = dict(schema="particle-slider-train-v1", profile="moss-cpu-fixture",
                  training=fixture_recipe())
    request = resolve_request(config, base=tmp_path, output=str(tmp_path / "first"),
                              no_reporting=True, checkpoint_every=1)
    checked = check_request(request)
    assert checked["qualification"] == "fixture"
    assert json.loads(checked["worker_output"])["model_loaded"] is False
    result = execute(request)
    assert result["status"] == "completed" and result["completed_steps"] == 2
    assert result["qualification"] == "fixture"
    packet = json.loads((Path(result["checkpoint"]) / "checkpoint.json").read_text())
    assert packet["pg_version"] == "0.9.5"
    adapter = json.loads((Path(result["adapter"]) / "slider.json").read_text())
    assert adapter["core_config"]["generator_arm"] == "adapter_gan"
    assert adapter["training"]["recipe"]["dummy"] is True
    resume = Path(result["checkpoint"]).with_name("step-000001")
    request = resolve_request(config, base=tmp_path, output=str(tmp_path / "resumed"),
                              no_reporting=True, resume=str(resume))
    replayed = execute(request)
    assert replayed["completed_steps"] == 2
    assert ((Path(replayed["adapter"]) / "adapter.safetensors").read_bytes() ==
            (Path(result["adapter"]) / "adapter.safetensors").read_bytes())
    # Invoke the merged user-facing dispatcher from outside the product tree.
    file = tmp_path / "request.json"
    file.write_text(json.dumps(config))
    command = [sys.executable, "-I", "-c",
               "from particle_sliders.session_cli import main; main()", "train",
               "--config", str(file), "--check", "--no-reporting"]
    completed = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["qualification"] == "fixture"
