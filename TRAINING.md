# Shared training frontend

Install the current product and exact shared core in a separate environment as
shown in README.md. The installed `moss` and `moss-cpu-fixture` profiles use
`particle-slider train`, the merged isolated execution API, and a product-owned
native worker/factory. Every GAN update still executes `ConditionalPG095`.

`moss` is source_only: CPU source and software checks do not qualify pretrained
CUDA/BF16/offload/media output. Its execution keeps the existing pinned model,
CUDA-only and local-files-only defaults. An allocation-free `--check` verifies
the recipe, complete resume-file presence and exact source/runtime closure; it
does not load a pretrained model or promise device/model readiness.

`moss-cpu-fixture` is fixture: it requires `dummy=true`, device cpu, at most 20
updates, rank 4, 64px, four rollout positions and batch 4. It uses a tiny random
native MOSS architecture; no model files are downloaded. It writes ordinary
shared-game full checkpoints and portable adapter exports labelled dummy=true.
The shared dispatcher owns process isolation and output/progress bookkeeping;
the product owns native model construction, GAN artifacts and source admission.
Native workers support periodic checkpoints and same-cohort resume. RAW adapter
selection is product owned. Training, evaluation, export and inference use raw
current weights without EMA state, checkpoint averaging or EMA comparisons.
This recipe has no host EMA option; do not add an unsupported `ema_decay`
field. EMA selection and export-format overrides are not advertised.
`--stop-after` is a session-only API and is not advertised here.

The old machine-local bridge on feat/unified-training-command-20261104 is a
separate source-bound experiment and is unchanged. Its provenance does not
qualify this portable factory. Historical source cohorts remain separate.

Current source uses core0.13.0/PG0.9.5 and product profile `hypergan-hndl-v3-pg095`. Old PG094 full-state/portable GAN packets require their original environment; fresh native qualification remains pending.
