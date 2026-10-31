# Draft reconciliation, 2026-10-31

This branch preserves existing PR1's shared GAN integration at
`dda9fbcf79a833fa4dc2185d0362f2da06980f02` and its exact core0.6.0 CDF/PG0.9.1
cohort. It replays the two dependency fixes already merged on public main:
`482a283` (Python>=3.11, Pillow>=13.0.0) and `ba24599` (Brotli1.2.1 hashes).
No GAN, native loader, sampler, adapter, checkpoint or old qualification receipt
has been rewritten. Core0.11/0.12 are not drop-in replacements: this draft imports
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
prior model/CPU/GPU qualifications. Publishing this reconciliation or adopting a
new UI/runtime cohort remains pending. Native libraries, caches, running campaigns,
original checkpoints and recorded lighting examples remain on their original pins.

The base runtime can now be resolved separately without the already optional UI.
See [runtime-only installation](runtime-only-install.md) and the new
requirements-runtime-cpu.lock. UI/full uv locking remains blocked; no runtime
behavior or native qualification changes follow from this packaging path.
