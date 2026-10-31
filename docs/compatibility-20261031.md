# Draft reconciliation, 2026-10-31

The initial security reconciliation at `51dd861` preserved PR1's shared GAN integration at
`dda9fbcf79a833fa4dc2185d0362f2da06980f02` and its exact core0.6.0 CDF/PG0.9.1
cohort. It replayed the two dependency fixes already merged on public main:
`482a283` (Python>=3.11, Pillow>=13.0.0) and `ba24599` (Brotli1.2.1 hashes).
That security-only stage changed no GAN, native loader, sampler, adapter,
checkpoint or old qualification receipt. Core0.11/0.12 are not drop-in replacements:
the historical cohort imports
`particle_sliders.pg09_moss_campaign`, absent from the clean current shared lane.

The combined UI dependency set is currently unsatisfiable. Actual `uv lock`
resolution rejects Pillow>=13.0.0 against Gradio6.30.0's Pillow<13 requirement.
Official latest Gradio6.31.0 also requires Pillow<13, so a simple version bump
cannot resolve it. Do not relax the patched image minimum, override the resolver,
or call the previous lock current. The inherited uv.lock and requirements-cpu.lock
retain historical package sets and cannot qualify this patched cohort. Brotli's
already merged fix is present, but the Pillow/Gradio conflict remains explicit.

Source wheel building can be checked independently with the declared Hatchling
backend. That is package construction only, not a complete install or re-run of
prior model/CPU/GPU qualifications. The reconciliation is published on draft PR1;
merge and adoption of a new UI/runtime cohort remain pending. Native libraries, caches, running campaigns,
original checkpoints and recorded lighting examples remain on their original pins.

The base runtime can now be resolved separately without the already optional UI.
See [runtime-only installation](runtime-only-install.md) and the new
requirements-runtime-cpu.lock. UI/full uv locking remains blocked; no runtime
behavior or native qualification changes follow from this packaging path.

Later source changes add the separate [HNDL preview](hypergan-hndl-preview.md)
and the reviewed [scheduler/graph ownership corrections](sft-scheduler-review-20261031.md).
Their installed CPU evidence is separate from the initial security-only checks;
historical checkpoint environments and original receipts remain preserved.
