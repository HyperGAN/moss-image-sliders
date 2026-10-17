"""Public checkpoint integrity through the actual shared MOSS GAN game."""

from copy import deepcopy
import hashlib
import json

import pytest
import torch

from moss_sliders.gan import restore_checkpoint, save_checkpoint
from test_gan import equal, fixture


@pytest.fixture(autouse=True)
def cpu():
    torch.set_num_threads(1)
    torch.manual_seed(17)


def rewrite_packet(directory, packet):
    """Keep file integrity valid so restore reaches optimizer-state validation."""
    path = directory / "training.pt"
    torch.save(packet, path)
    metadata_path = directory / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))


@pytest.mark.parametrize("optimizer", ["opt_g", "opt_d"])
@pytest.mark.parametrize("corruption", [
    "negative_step", "nonfinite_step", "missing_step", "extra_field",
    "deleted_history", "empty_history", "cleared_histories",
])
def test_started_moss_checkpoint_rejects_corrupt_history_without_mutation(
        tmp_path, optimizer, corruption):
    make, _, _ = fixture()
    source = make()
    source.step()
    source.step()
    directory = save_checkpoint(source, tmp_path / "source")
    packet = source.game.state_dict()
    history = packet[optimizer]["state"]
    key = next(iter(history))
    if corruption == "negative_step":
        history[key]["step"] = -1
    elif corruption == "nonfinite_step":
        history[key]["step"] = float("nan")
    elif corruption == "missing_step":
        del history[key]["step"]
    elif corruption == "extra_field":
        history[key]["unexpected"] = 1
    elif corruption == "deleted_history":
        del history[key]
    elif corruption == "empty_history":
        history[key] = {}
    else:
        history.clear()
    # Preserve the checkpoint's recorded history schema, including the used D
    # parameters. This is loss of saved state, rather than an unused critic leaf.
    rewrite_packet(directory, packet)

    receiver = make()
    receiver.step()
    before = receiver.game.state_dict()
    control = make()
    control.game.load_state_dict(deepcopy(before))
    expected_metric = control.step()
    expected_sample = control.game.evaluate(0, seed=93)

    with pytest.raises(ValueError):
        restore_checkpoint(receiver, directory)
    equal(before, receiver.game.state_dict())
    assert receiver.step() == expected_metric
    equal(control.game.state_dict(), receiver.game.state_dict())
    assert torch.equal(expected_sample, receiver.game.evaluate(0, seed=93))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("completed_steps", [0, 2])
def test_valid_moss_checkpoint_preserves_exact_update_and_sample(tmp_path, completed_steps):
    make, _, _ = fixture()
    source = make()
    for _ in range(completed_steps):
        source.step()
    snapshot = source.game.state_dict()
    if completed_steps == 0:
        assert snapshot["opt_g"]["state"] == snapshot["opt_d"]["state"] == {}
    directory = save_checkpoint(source, tmp_path / "source")
    expected_metric = source.step()
    expected_sample = source.game.evaluate(0, seed=93)

    receiver = make()
    assert restore_checkpoint(receiver, directory) == completed_steps
    equal(snapshot, receiver.game.state_dict())
    assert receiver.step() == expected_metric
    equal(source.game.state_dict(), receiver.game.state_dict())
    assert torch.equal(expected_sample, receiver.game.evaluate(0, seed=93))
    assert not torch.cuda.is_initialized()
