"""Base packaging and CLI discovery do not require the incompatible UI extra."""
from pathlib import Path
import re
import importlib.util

import pytest
import subprocess
import sys
import tomllib

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def test_ui_is_optional_and_runtime_lock_keeps_patched_numeric_cohort():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    base = {Requirement(value).name.lower() for value in project['dependencies']}
    assert 'gradio' not in base
    assert project['optional-dependencies']['ui'] == ['gradio==6.30.0']
    assert 'Pillow>=13.0.0' in project['dependencies']
    assert any('cdfb80a76b882b4b73861b12f555263bd64ffeb7' in value for value in project['dependencies'])
    text = (ROOT / 'requirements-runtime-cpu.lock').read_text()
    assert 'pillow==13.0.0' in text
    assert 'particlegan==0.9.1' in text
    assert 'torch @ https://download-r2.pytorch.org/whl/cpu/torch-2.9.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl' in text
    assert '--index-url https://pypi.org/simple' in text
    assert '--extra-index-url' not in text
    entries = re.findall(r'^([A-Za-z0-9][A-Za-z0-9_.-]*)(?:==| @ )', text, flags=re.M)
    assert 'gradio' not in entries and 'gradio-client' not in entries
    assert not any(name.startswith('nvidia-') for name in entries)
    # The VCS core has its own exact-commit installation; all entries in this
    # artifact can therefore use normal pip --require-hashes validation.
    assert 'git+' not in text
    assert 'particle-sliders-core==' not in text
    for entry in re.split(r'\n(?=[A-Za-z0-9][A-Za-z0-9_.-]*(?:==| @ ))', text):
        if re.search(r'^[A-Za-z0-9][A-Za-z0-9_.-]*(?:==| @ )', entry, flags=re.M):
            assert re.search(r'--hash=sha256:[0-9a-f]{64}', entry)


def test_installed_base_cli_help_never_imports_gradio(tmp_path):
    code = r'''
import importlib.abc
from importlib.metadata import distribution
from pathlib import Path
import sys
class NoUI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'gradio' or fullname.startswith('gradio.'):
            raise AssertionError('base CLI imported the optional UI')
sys.meta_path.insert(0, NoUI())
dist = distribution('moss-image-sliders')
entry = next(ep for ep in dist.entry_points if ep.name == 'moss-sliders')
main = entry.load()
assert 'site-packages' in Path(sys.modules['moss_sliders.cli'].__file__).parts
sys.argv = ['moss-sliders', '--help']
try:
    main()
except SystemExit as stopped:
    assert stopped.code == 0
else:
    raise AssertionError('help must exit without model execution')
assert not any(name == 'gradio' or name.startswith('gradio.') for name in sys.modules)
assert 'moss_sliders.backend' not in sys.modules
assert 'moss_sliders.gan' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-I', '-c', code], cwd=tmp_path,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert all(name in result.stdout for name in ('train', 'infer', 'ui', 'smoke'))


def test_torch_direct_archive_hash_is_exact_and_cannot_be_silently_rebound():
    spec = importlib.util.spec_from_file_location('moss_lock_tool', ROOT / 'tools/lock_runtime_cpu.py')
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    url = next(line.removeprefix('torch @ ') for line in
               (ROOT / 'requirements-runtime-cpu.in').read_text().splitlines()
               if line.startswith('torch @ '))
    text = (ROOT / 'requirements-runtime-cpu.lock').read_text()
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('torch @ '))
    assert lines[start + 1].strip() == '--hash=sha256:' + url.split('#sha256=')[1]
    assert not lines[start + 2].lstrip().startswith('--hash=')
    polluted = text.replace(lines[start + 1], '    --hash=sha256:' + '0' * 64 + ' ' + chr(92) + chr(10) + lines[start + 1], 1)
    assert tool.bind_torch_archive(polluted, url) == text
    with pytest.raises(ValueError, match='differs from the platform input'):
        tool.bind_torch_archive(text, url.replace('2.9.1', '2.9.2'))
    with pytest.raises(ValueError, match='omit the published'):
        tool.bind_torch_archive(text.replace(lines[start + 1], '    --hash=sha256:' + '0' * 64, 1), url)


def test_hndl_profile_refuses_historical_runtime_before_native_attachment():
    from importlib.metadata import version
    if version('particle-sliders-core') != '0.6.0':
        pytest.skip('historical runtime admission check belongs to the base environment')
    from moss_sliders.native_hndl import build_hndl_run
    with pytest.raises(ValueError, match='separately qualified core'):
        build_hndl_run(object(), (), {})


def test_public_hndl_cpu_profile_pins_exact_source_and_keeps_archive_hashes():
    project = tomllib.loads((ROOT / 'profiles/hypergan-hndl/pyproject.toml').read_text())['project']
    core = next(Requirement(value) for value in project['dependencies']
                if Requirement(value).name == 'particle-sliders-core')
    assert core.extras == {'serialization', 'hypergan-hndl'}
    assert core.url == ('git+https://github.com/HyperGAN/particle-sliders.git@'
                        '4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231'
                        '#subdirectory=packages/particle-sliders-core')
    text = (ROOT / 'requirements-hndl-cpu.lock').read_text()
    names = re.findall(r'^([A-Za-z0-9][A-Za-z0-9_.-]*)(?:==| @ )', text, flags=re.M)
    assert len(names) == 45
    assert 'pillow==13.0.0' in text and 'particlegan==0.9.4' in text and 'hndl==0.10.0' in text
    assert 'gradio' not in names and not any(name.startswith('nvidia-') for name in names)
    assert 'git+' not in text and '--extra-index-url' not in text
    for entry in re.split(r'\n(?=[A-Za-z0-9][A-Za-z0-9_.-]*(?:==| @ ))', text):
        if re.search(r'^[A-Za-z0-9][A-Za-z0-9_.-]*(?:==| @ )', entry, flags=re.M):
            assert re.search(r'--hash=sha256:[0-9a-f]{64}', entry)
