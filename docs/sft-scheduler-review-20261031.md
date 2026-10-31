# SFT scheduler and review corrections

The model is **OpenMOSS-Team/MOSS-Image-SFT**, revision
`fc5de0d56fc910cbca6f6625a70f5f582b26deb1`. The pinned upstream
[model index](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/raw/fc5de0d56fc910cbca6f6625a70f5f582b26deb1/model_index.json)
names `MOSSImageScheduler`; its
[scheduler config](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/raw/fc5de0d56fc910cbca6f6625a70f5f582b26deb1/scheduler/scheduler_config.json)
has 1000 training timesteps, shift5, training_shift1 and dynamic shifting disabled.
The pinned upstream pipeline source is byte-identical to the vendored file,
SHA256 `1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31`.
Only these small configuration/source files were fetched; no weights were fetched.

Running that installed scheduler gives the four-step grid
`[1000, 909.9999389648438, 717, 24]`, approximately `[1000, 910, 717, 24]`.
The review's suggested grid is therefore **already produced by this SFT code**;
no new Turbo schedule or hard-coded grid was adopted. The 50-step grid contains
50 distinct timesteps from1000 to24; both complete grids and sigmas are recorded
in [the review receipt](receipts/review-fixes-20261031/scheduler-grids.json).

The current example recipes use four-step frozen neutral training rollouts,
while the rendering example uses50 steps. The preserved historical lighting
[provenance](../samples/lighting-warmth/provenance.json) also records rollout_steps4,
inference_steps50, shift5 and this SFT revision. These are different numerical
trajectories. CPU integration tests do not establish pretrained quality or
robustness across their timesteps; that remains an explicit native qualification
gap. Neither the render budget nor the training recipe was changed on the basis
of the review suggestion.

One concrete scheduler defect was fixed: `sample_training_state` previously reset
and advanced the caller's scheduler. It now constructs a fresh scheduler from the
same config, as rendering already does. Tests preserve an in-progress50-step
scheduler's full state and object identity on success and model-forward failure,
and verify bit-exact agreement with the prior four-step numerical trajectory.

## Adapter and graph ownership

Both MOSS cohorts are **adapter GANs**. The shared game receives `particles=None`,
has `prior is None` and records `generator_arm: adapter_gan`. Only the signed
attention LoRA leaves train. The preview's `hypergan-hndl-v2` identity names the
verified builder/runtime cohort; it does not turn the LoRA into Fibo's routed
128x4-bank architecture. Its two-linear HNDL LoRA graph, native shapes, scale
factor and source hash are recorded separately. No bank was added, and strength1
and the historical signed ramp recommendations were retained.

The HNDL graph shares the exact registered `down`/`up` modules with the residual,
so there are two parameter leaves and exactly `down.weight`/`up.weight` state keys.
The graph is intentionally stored outside the module registry to avoid duplicate
portable keys. Review testing found that converting only the registered leaves
could leave HNDL's runtime device/dtype contract stale. The residual now delegates
conversion through HNDL's `_apply` hook once and propagates train/eval state.
Tests cover shared parameter identity, no hidden graph buffers, deepcopy,
float32→float64→float32 execution, state reload and CPU→meta→CPU relocation.
Meta is a state-contract test, not CUDA/BF16 qualification.

Full checkpoint packets explicitly retain PG0.9.4, `release-0.9.4`, penalty domain,
core commit and verified network-runtime identity. Changing the PG version,
source profile or penalty domain rejects without receiver mutation. The complete
packaged core Python/JSON/HNDL closure rejects changed code, graph or data and
unexpected extra Python files before hooks; the builder's HNDL/HyperGAN closure
checks remain active. Historical receipts and checkpoint cohorts are preserved.
