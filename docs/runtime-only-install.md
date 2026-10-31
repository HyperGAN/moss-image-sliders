# Base runtime without the optional UI

Gradio is already isolated in the `ui` extra and imported only when building
that interface. The new `requirements-runtime-cpu.lock` resolves the base runtime
without UI dependencies for Python3.12/Linux x86-64/CPU. It keeps Pillow13.0.0,
Torch2.9.1+cpu, Transformers5.3.0 and the exact PG0.9.1 dependencies. This lock is
not a CUDA environment or native/pretrained model qualification.

From the source repository root in a new, separate Python3.12 environment:

```bash
python -m pip install --require-hashes -r requirements-runtime-cpu.lock
python -m pip install --no-deps 'particle-sliders-core[serialization] @ git+https://github.com/HyperGAN/particle-sliders.git@cdfb80a76b882b4b73861b12f555263bd64ffeb7#subdirectory=packages/particle-sliders-core'
python -m pip install --no-deps .
moss-sliders --help
```

The Git package is intentionally a separate exact-commit step: pip's archive-hash
mode cannot hash a VCS requirement. Its transitive dependencies are included in
the runtime lock, and existing runtime source/checkpoint identities still apply.
No global environment or active native campaign should be retargeted. This path
has dependency-resolution, wheel-build and installed CLI-discovery evidence; a
complete fresh native installation or model execution was not performed here.

Do not add `.[ui]`, use the historical all-extras lock as a patched environment,
or lower Pillow to satisfy it. Gradio6.30 and latest6.31 require Pillow<13; the
optional UI and universal `uv lock` remain blocked until compatible dependencies
exist. The existing `ui` extra, application code, rendering and training behavior
are unchanged. No server is launched by these packaging checks.

The command header in the lock records reproducible generation from pyproject,
using `--no-emit-package particle-sliders-core` to keep the exact VCS step separate.
Historical `uv.lock` and `requirements-cpu.lock` retain their cohort meaning.

When regenerating, retain the explicit PyTorch CPU index line. uv resolves
`--torch-backend cpu` dynamically but did not emit that index in this tool version;
the lock therefore declares the same resolved official artifact source for pip.
All resolved archive requirements remain hash-checked.
