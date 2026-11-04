"""Local bridge finalization rejects incomplete or changed native artifacts."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'tools/train_local_pg.py'
SPEC = importlib.util.spec_from_file_location('moss_local_bridge', PATH)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


def completed(root):
    bridge.write(root / 'result.json', {'status': 'completed', 'completed_steps': 128})
    bridge.write(root / 'launcher.json', {'exit_code': 0, 'worker_reaped': True})
    checkpoint = root / 'checkpoints/step-000128'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'training.pt').write_bytes(b'checkpoint')
    bridge.write(checkpoint / 'checkpoint.json', {'step': 128, 'sha256': bridge.sha(checkpoint / 'training.pt')})
    adapter = root / 'adapter'
    adapter.mkdir()
    (adapter / 'adapter.safetensors').write_bytes(b'adapter')
    bridge.write(adapter / 'slider.json', {'completed_steps': 128, 'adapter_sha256': bridge.sha(adapter / 'adapter.safetensors')})


def test_complete_result_references_native_checkpoint(tmp_path):
    completed(tmp_path)
    result = bridge.verify(tmp_path, 128)
    assert result['checkpoint'] == str(tmp_path / 'checkpoints/step-000128')
    assert result['adapter'] == str(tmp_path / 'adapter')


def test_changed_adapter_cannot_complete(tmp_path):
    completed(tmp_path)
    (tmp_path / 'adapter/adapter.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='adapter bytes'):
        bridge.verify(tmp_path, 128)


def test_unclean_native_launcher_cannot_complete(tmp_path):
    completed(tmp_path)
    bridge.write(tmp_path / 'launcher.json', {'exit_code': 9, 'worker_reaped': True})
    with pytest.raises(ValueError, match='finish cleanly'):
        bridge.verify(tmp_path, 128)
