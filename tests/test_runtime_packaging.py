"""Base packaging and CLI discovery do not require the incompatible UI extra."""
from pathlib import Path
import re
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
    assert 'torch==2.9.1+cpu' in text
    assert 'https://download.pytorch.org/whl/cpu' in text
    entries = re.findall(r'^([A-Za-z0-9][A-Za-z0-9_.-]*)==', text, flags=re.M)
    assert 'gradio' not in entries and 'gradio-client' not in entries
    assert not any(name.startswith('nvidia-') for name in entries)
    # The VCS core has its own exact-commit installation; all entries in this
    # artifact can therefore use normal pip --require-hashes validation.
    assert 'git+' not in text
    assert 'particle-sliders-core==' not in text
    for entry in re.split(r'\n(?=[A-Za-z0-9][A-Za-z0-9_.-]*==)', text):
        if re.search(r'^[A-Za-z0-9][A-Za-z0-9_.-]*==', entry, flags=re.M):
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
