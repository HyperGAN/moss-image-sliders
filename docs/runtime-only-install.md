> Historical cohort/evidence. Current package installation and training are described in [PG0.9.4 current](pg094-current.md). Original receipts below remain unchanged.

# Base runtime without the optional UI

Gradio is already isolated in the `ui` extra and imported only when building
that interface. The new `requirements-runtime-cpu.lock` resolves the base runtime
without UI dependencies for Python 3.12/Linux x86-64/CPU. It keeps Pillow 13.0.0,
Torch 2.9.1+cpu, Transformers5.3.0 and the exact PG 0.9.1 dependencies. This lock is
not a CUDA environment or native/pretrained model qualification.

From the source repository root in a new, separate Python 3.12 environment:

```bash
python -m pip install --require-hashes -r requirements-runtime-cpu.lock
python -m pip install --no-deps 'particle-sliders-core[serialization] @ git+https://github.com/HyperGAN/particle-sliders.git@cdfb80a76b882b4b73861b12f555263bd64ffeb7#subdirectory=packages/particle-sliders-core'
python -m pip install --no-deps .
moss-sliders --help
```

The Git package is intentionally a separate exact-commit step: pip's archive-hash
mode cannot hash a VCS requirement. Its transitive dependencies are included in
the runtime lock, and existing runtime source/checkpoint identities still apply.
No global environment or active native campaign should be retargeted. The
2026-10-31 follow-up completed this base installation in a fresh isolated
Python 3.12.15 environment with actual Torch 2.9.1+cpu and Pillow 13, built/installed
the pinned core and product wheels, and passed 87 base CPU tests. `pip check`
passes. Seven separate HNDL-profile tests are skipped in this base environment;
the incompatible optional UI test is explicitly deselected. Raw installed-wheel
receipts are in [runtime/HNDL evidence](receipts/runtime-hndl-20261031/receipt.json).
This is still not pretrained/native GPU or image-quality qualification.

Do not add `.[ui]`, use the historical all-extras lock as a patched environment,
or lower Pillow to satisfy it. Gradio 6.30 and latest 6.31 require Pillow<13; the
optional UI and universal `uv lock` remain blocked until compatible dependencies
exist. The existing `ui` extra, application code, rendering and training behavior
are unchanged. No server is launched by these packaging checks.

The command header in the lock records reproducible generation from pyproject,
using `--no-emit-package particle-sliders-core` to keep the exact VCS step separate.
Historical `uv.lock` and `requirements-cpu.lock` retain their cohort meaning.

`requirements-runtime-cpu.in` binds the exact official CPython 3.12/Linux CPU
Torch archive and its published SHA256. Other packages remain on PyPI, so no
broad extra-index search or relaxed uv index policy is needed. The lock header
records generation from both pyproject.toml and this platform-specific input.
All emitted archive requirements remain hash-checked.

Regenerate with `python tools/lock_runtime_cpu.py --uv /path/to/uv`. The tool
requires the official input hash to appear in resolved hashes, then narrows
uv release-wide Torch hashes to that one explicitly named archive. It does not
turn off hash validation, relax index policy, or change package versions.
