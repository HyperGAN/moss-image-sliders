# MOSS correction evidence and complete file index

This index makes the correction handoff readable in the public PR. The earlier
Claude correction source is pinned at
[`d21da731eb2504c8ba42a54810adb071624ae0a4`](https://github.com/HyperGAN/moss-image-sliders/commit/d21da731eb2504c8ba42a54810adb071624ae0a4).
The final bounded delta adds only public-candidate installation plumbing,
checkpoint-header preflight, the four-step diagnostic example, documentation
spacing and independent receipts. The default 50-step render and strengths stay
unchanged. Relative source/test links below resolve at this document's commit.

## Issue-by-issue evidence

1. **Install/docs:** [README at the correction head](https://raw.githubusercontent.com/HyperGAN/moss-image-sliders/d21da731eb2504c8ba42a54810adb071624ae0a4/README.md)
   replaced stale all-extras/old-lock instructions with the 43-archive base CPU
   lock, exact core `--no-deps`, then product `--no-deps`. Eligibility is Python
   3.11–3.12; the lock qualifies Python 3.12/Linux x86-64. A third fresh environment
   executed that sequence, `pip check`, help and tiny CPU smoke. Current
   [preview install instructions](hypergan-hndl-preview.md) independently close
   the public core candidate's clean CPU installation blocker.
2. **Actual scheduler:** [pinned SFT source/grid review](sft-scheduler-review-20261031.md)
   records byte-identical upstream/vendor pipeline SHA `1dac1aca...`, shift 5,
   training shift 1, the existing four-step grid `[1000, 909.9999389648438, 717, 24]`
   and the full distinct 50-step grid. Historical provenance already records
   rollout 4/render 50. No mandatory four-step render or 2x strength was adopted.
   README offers `--steps 4` only as a diagnostic grid reproduction.
3. **Borrowed scheduler:** [actual clone source](https://raw.githubusercontent.com/HyperGAN/moss-image-sliders/d21da731eb2504c8ba42a54810adb071624ae0a4/src/moss_sliders/training.py)
   and [regressions](../tests/test_training_scheduler.py) preserve the caller's
   full in-progress scheduler state and object identity on success/error,
   with bit-exact prior numerical trajectory:

   ```python
   scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
   scheduler.set_timesteps(rollout_steps, device=pipe.transformer.device)
   ```

4. **Graph ownership/conversion:** [actual binding source](https://raw.githubusercontent.com/HyperGAN/moss-image-sliders/d21da731eb2504c8ba42a54810adb071624ae0a4/src/moss_sliders/native_hndl.py)
   fixes stale HNDL device/dtype contracts by forwarding conversion through the
   graph once while preserving the two portable registered leaves:

   ```python
   if recurse:
       self._graph._apply(fn, recurse=True)
       return self
   ```

   [Tests](../tests/test_native_hndl.py) check graph/registered parameter identity,
   only `down.weight`/`up.weight` state, no hidden buffers, deepcopy, float64/float32
   roundtrip, train/eval and CPU/meta/CPU reload. Meta is not CUDA qualification.
5. **Adapter semantics:** the actual game receives `particles=None`, has
   `prior is None` and exports `generator_arm=adapter_gan`. The HNDL builder cohort
   does not imply Fibo's routed 128x4-bank architecture. No bank or strength was
   invented. Real HNDL LoRA VJPs, shared critic second derivatives, game ownership,
   exact resume, zero strength and actual-weight reload are tested.
6. **Checkpoint preflight and source closure:** [source](../src/moss_sliders/gan.py)
   writes `moss-pg09-checkpoint-v2` JSON containing `pg_version`,
   `pg_source_profile`, `penalty_input_domain`. Mismatches or incomplete headers
   reject before `torch.load`; old unidentified v1 headers reject before tensor
   load in the preview, while original PG 0.9.1 v1 resume still passes.
   [Tests](../tests/test_native_hndl.py) change header identities with tensor
   loading forbidden. The separately labelled [actual old 0.9.1 external receipt](receipts/public-cpu-candidate-20261031/actual-old091-preload-rejection.json)
   records zero tensor loads and unchanged receiver. Full core Python/JSON/HNDL
   closure changes and unexpected extra source files reject before hooks.
7. **Public core and clean CPU install:** exact core
   [`4abc0eb`](https://github.com/HyperGAN/particle-sliders/commit/4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231)
   is independently verified at draft PR148; all 49 admitted files match the
   unchanged producer manifest. The public Git build, not a borrowed local core
   wheel, is installed with exact HyperGAN `5f368a`, HNDL 0.10 and official PG 0.9.4.
   The separate 45-archive hash lock retains Pillow 13 and exact CPU Torch.
   [Public provenance](receipts/public-cpu-candidate-20261031/public-core-provenance.json),
   [VCS/build/source closure](receipts/public-cpu-candidate-20261031/vcs-and-source-closure.json)
   and [clean pip check](receipts/public-cpu-candidate-20261031/pip-check.txt) are public.

## Final bounded validation

[Exact commands/source hashes and results](receipts/public-cpu-candidate-20261031/receipt.json):
**23 candidate installed CPU cases passed**, no skips; **92 base cases passed**,
19 HNDL-only skips and one incompatible UI test explicitly deselected. The fresh
public-source CLI/help/four-update smoke, both pip checks, source admission,
Ruff and diff checks pass. Historical and base locks plus the immutable checkpoint
source manifest are unchanged. Prior dated receipts keep their original scope.

Source availability and clean CPU installation are qualified. UI remains blocked
by Pillow >=13 versus Gradio <13; pretrained/GPU/BF16/offload/image quality,
memory and performance remain unrun. Core PR148 is a draft candidate, not a merged
stable release. No model downloads, releases, deployments or merges occurred.

## Complete file lists

### Claude correction delta: 05a39ce → d21da73

22 files, including this index where applicable.

```text
README.md
docs/compatibility-20261031.md
docs/model-integration.md
docs/receipts/review-fixes-20261031/base-installed-provenance.json
docs/receipts/review-fixes-20261031/base-installed-tests.xml
docs/receipts/review-fixes-20261031/cross-cohort-checkpoint.txt
docs/receipts/review-fixes-20261031/hndl-installed-provenance.json
docs/receipts/review-fixes-20261031/hndl-installed-tests.xml
docs/receipts/review-fixes-20261031/hndl-smoke.json
docs/receipts/review-fixes-20261031/readme-environment.json
docs/receipts/review-fixes-20261031/readme-pip-check.txt
docs/receipts/review-fixes-20261031/readme-smoke.json
docs/receipts/review-fixes-20261031/receipt.json
docs/receipts/review-fixes-20261031/scheduler-grids.json
docs/receipts/review-fixes-20261031/upstream-sft-source.json
docs/sft-scheduler-review-20261031.md
docs/shared-pg09-audit.md
docs/validation.md
src/moss_sliders/native_hndl.py
src/moss_sliders/training.py
tests/test_native_hndl.py
tests/test_training_scheduler.py
```

### Final bounded delta: d21da73 → this commit

25 files, including this index where applicable.

```text
README.md
docs/compatibility-20261031.md
docs/hypergan-hndl-preview.md
docs/receipts/public-cpu-candidate-20261031/actual-old091-preload-rejection.json
docs/receipts/public-cpu-candidate-20261031/actual-old091-provenance.json
docs/receipts/public-cpu-candidate-20261031/base-installed-tests.xml
docs/receipts/public-cpu-candidate-20261031/base-pip-check.txt
docs/receipts/public-cpu-candidate-20261031/installed-provenance.json
docs/receipts/public-cpu-candidate-20261031/installed-tests.xml
docs/receipts/public-cpu-candidate-20261031/pip-check.txt
docs/receipts/public-cpu-candidate-20261031/public-core-provenance.json
docs/receipts/public-cpu-candidate-20261031/receipt.json
docs/receipts/public-cpu-candidate-20261031/remote-core-ref.txt
docs/receipts/public-cpu-candidate-20261031/smoke.json
docs/receipts/public-cpu-candidate-20261031/vcs-and-source-closure.json
docs/review-corrections-index.md
docs/runtime-only-install.md
docs/sft-scheduler-review-20261031.md
profiles/hypergan-hndl/pyproject.toml
requirements-hndl-cpu.lock
src/moss_sliders/gan.py
tests/test_gan.py
tests/test_native_hndl.py
tests/test_runtime_packaging.py
tools/lock_runtime_cpu.py
```

### Entire PR: main 38d856f → this commit

99 files, including this index where applicable.

```text
README.md
docs/campaign.md
docs/compatibility-20261031.md
docs/hypergan-hndl-preview.md
docs/model-integration.md
docs/receipts/public-cpu-candidate-20261031/actual-old091-preload-rejection.json
docs/receipts/public-cpu-candidate-20261031/actual-old091-provenance.json
docs/receipts/public-cpu-candidate-20261031/base-installed-tests.xml
docs/receipts/public-cpu-candidate-20261031/base-pip-check.txt
docs/receipts/public-cpu-candidate-20261031/installed-provenance.json
docs/receipts/public-cpu-candidate-20261031/installed-tests.xml
docs/receipts/public-cpu-candidate-20261031/pip-check.txt
docs/receipts/public-cpu-candidate-20261031/public-core-provenance.json
docs/receipts/public-cpu-candidate-20261031/receipt.json
docs/receipts/public-cpu-candidate-20261031/remote-core-ref.txt
docs/receipts/public-cpu-candidate-20261031/smoke.json
docs/receipts/public-cpu-candidate-20261031/vcs-and-source-closure.json
docs/receipts/review-fixes-20261031/base-installed-provenance.json
docs/receipts/review-fixes-20261031/base-installed-tests.xml
docs/receipts/review-fixes-20261031/cross-cohort-checkpoint.txt
docs/receipts/review-fixes-20261031/hndl-installed-provenance.json
docs/receipts/review-fixes-20261031/hndl-installed-tests.xml
docs/receipts/review-fixes-20261031/hndl-smoke.json
docs/receipts/review-fixes-20261031/readme-environment.json
docs/receipts/review-fixes-20261031/readme-pip-check.txt
docs/receipts/review-fixes-20261031/readme-smoke.json
docs/receipts/review-fixes-20261031/receipt.json
docs/receipts/review-fixes-20261031/scheduler-grids.json
docs/receipts/review-fixes-20261031/upstream-sft-source.json
docs/receipts/runtime-hndl-20261031/base-installed-provenance.json
docs/receipts/runtime-hndl-20261031/base-installed-tests.xml
docs/receipts/runtime-hndl-20261031/cross-cohort-checkpoint.txt
docs/receipts/runtime-hndl-20261031/hndl-installed-provenance.json
docs/receipts/runtime-hndl-20261031/hndl-installed-tests.xml
docs/receipts/runtime-hndl-20261031/hndl-smoke.json
docs/receipts/runtime-hndl-20261031/receipt.json
docs/receipts/runtime-hndl-20261031/ui-resolution.txt
docs/receipts/shared-pg09-audit/active-source-baseline.json
docs/receipts/shared-pg09-audit/active-source-check.json
docs/receipts/shared-pg09-audit/consumer-source-tests.xml
docs/receipts/shared-pg09-audit/cpu-gan-smoke.json
docs/receipts/shared-pg09-audit/installed-wheel-smoke.json
docs/receipts/shared-pg09-audit/installed-wheel-tests.xml
docs/receipts/shared-pg09-audit/legacy-defects.json
docs/receipts/shared-pg09-audit/probe_legacy_defects.py
docs/receipts/shared-pg09-audit/receipt.json
docs/receipts/shared-pg09-cdf/active-source-check.json
docs/receipts/shared-pg09-cdf/installed-wheel-smoke.json
docs/receipts/shared-pg09-cdf/installed-wheel-tests.xml
docs/receipts/shared-pg09-cdf/receipt.json
docs/receipts/shared-pg09-cdf/release-ref-baseline.json
docs/receipts/shared-pg09-replacement/active-source-check.json
docs/receipts/shared-pg09-replacement/checkpoint-integrity-tests.xml
docs/receipts/shared-pg09-replacement/installed-wheel-smoke.json
docs/receipts/shared-pg09-replacement/installed-wheel-tests.xml
docs/receipts/shared-pg09-replacement/receipt.json
docs/receipts/shared-pg09-replacement/release-ref-baseline.json
docs/receipts/shared-pg09-replacement/rng-scope-baseline.json
docs/review-corrections-index.md
docs/runtime-only-install.md
docs/sft-scheduler-review-20261031.md
docs/shared-pg09-audit.md
docs/validation.md
examples/warm-light.json
examples/watercolor.json
profiles/hypergan-hndl/pyproject.toml
pyproject.toml
requirements-cpu.lock
requirements-hndl-cpu.lock
requirements-runtime-cpu.in
requirements-runtime-cpu.lock
src/moss_sliders/artifacts.py
src/moss_sliders/backend.py
src/moss_sliders/campaign.py
src/moss_sliders/cli.py
src/moss_sliders/gan.py
src/moss_sliders/hndl_cohort.json
src/moss_sliders/inference.py
src/moss_sliders/lora.py
src/moss_sliders/native_hndl.py
src/moss_sliders/smoke.py
src/moss_sliders/training.py
tests/test_campaign.py
tests/test_checkpoint_replacement.py
tests/test_gan.py
tests/test_inference_export_edges.py
tests/test_inference_pg09.py
tests/test_integration.py
tests/test_native_hndl.py
tests/test_rng_scope.py
tests/test_runtime_packaging.py
tests/test_sliders.py
tests/test_training_scheduler.py
tools/build_hndl_profile.py
tools/cuda_campaign_smoke.py
tools/lock_runtime_cpu.py
tools/monitor_campaign.py
tools/train_campaign.py
uv.lock
```
