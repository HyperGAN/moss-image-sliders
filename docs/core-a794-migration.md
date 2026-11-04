> Historical cohort/evidence. Current package installation and training are described in [PG0.9.4 current](pg094-current.md). Original receipts below remain unchanged.

# MOSS modern core migration: Claude RETURN packet

The active modern preview alone moves from core `4abc0eb` to exact
`a794e3b4a8813b012f2c4faa3da82b78d4a90214`. Product identity becomes
`moss-image-sliders-hndl 0.3.0.dev1`. Default core 0.6.0/PG 0.9.1, historical
locks, licenses, objectives, attention-LoRA architecture, model gates and
checkpoint JSON behavior remain unchanged. Earlier review notes and receipts
remain preserved at their recorded heads.

## Admission and artifact identity

[Exact dependency](../profiles/hypergan-hndl/pyproject.toml),
[recursive manifest](../src/moss_sliders/hndl_cohort.json), and
[admission source](../src/moss_sliders/native_hndl.py) bind the new cohort to
all 50 core package source/data files plus the compatibility namespace file.
The old manifest is preserved in `profiles/hypergan-hndl/history/`.

```python
actual = closure(root)
if actual != cohort["sources"]:
    raise ValueError("MOSS HNDL core source closure differs from the exact local candidate")
```

The source/profile/manifest identity changes deliberately. A real old `4abc`
full packet and portable export reject with receiver state unchanged; neither
packet nor its metadata is relabelled. Resume old packets using product
`fc1dad414ed0b98666528cc4d8672a629ea072b8` and its exact original core. New
artifacts are a separate cohort, despite core retaining version 0.12.0.

The old-artifact check is a **separate external probe**, not one of the 24 pytest
cases. [Producer provenance, saved hashes, exact generation/refusal commands and
output](receipts/core-a794-migration/old-artifact-provenance.json) identify the
retained `fc1dad4` wheel and `4abc` core. These are controlled tiny CPU fixtures
generated before migration, not production or pretrained checkpoints. Their
metadata and payloads were not relabelled.

## Installed CPU evidence

The exact VCS dependency was resolved and installed in a fresh isolated CPU
environment. The wheel and sdist were built, then installed/tested separately.
The test bundle contains no product checkout/src. An isolated `python -I`
origin probe verifies site-packages imports and every product wheel/source/
installed byte, exact core Git provenance and complete recursive admission.
Both pip checks pass.

- Wheel: 24 passed, zero skipped.
- Sdist rebuilt install: the same 24 passed, zero skipped.
- Raw adapter updates/ownership, OFF/zero, export/reload, exact resume, strict
  source/header refusals, late-build rollback and same-pipeline retry pass.
- Four-update tiny-native CPU CLI smoke passes from `/tmp`.
- MOSS exposes raw adapter/game evaluation, not EMA selection. The shared core's
  EMA qualification does not establish a product EMA API; none is claimed here.

[Receipt with exact commands and artifact hashes](receipts/core-a794-migration/receipt.json),
[installed origins](receipts/core-a794-migration/installed-origin.json),
[old packet/export refusal](receipts/core-a794-migration/old-cohort-rejection.json),
[wheel tests](receipts/core-a794-migration/wheel-tests.xml),
[sdist tests](receipts/core-a794-migration/sdist-tests.xml).

The parent reported bounded synthetic CUDA qualification of the exact shared
core; this owner ran no GPU job and claims only CPU product software evidence.
UI's patched-Pillow/Gradio conflict and pretrained MOSS/GPU/BF16/offload/media
quality/performance gates remain unqualified. No widths/domains were admitted,
no release/deployment/workflow/access changes occurred. Root must complete
Claude actual-diff RETURN review before any merge.

## Complete changed-file index since prior product head fc1dad4

```text
docs/core-a794-migration.md
docs/hypergan-hndl-preview.md
docs/receipts/core-a794-migration/installed-origin.json
docs/receipts/core-a794-migration/old-cohort-rejection.json
docs/receipts/core-a794-migration/pip-check-sdist.txt
docs/receipts/core-a794-migration/receipt.json
docs/receipts/core-a794-migration/sdist-tests.xml
docs/receipts/core-a794-migration/smoke.json
docs/receipts/core-a794-migration/wheel-tests.xml
profiles/hypergan-hndl/history/core-4abc-cohort.json
profiles/hypergan-hndl/pyproject.toml
src/moss_sliders/hndl_cohort.json
src/moss_sliders/native_hndl.py
tests/test_native_hndl.py
tests/test_runtime_packaging.py
tools/build_hndl_profile.py
```
