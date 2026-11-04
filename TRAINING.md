# One training command

```sh
./train --config training.example.json
```

`./train` forwards to the common `particle-slider train` command with this
project's explicit `training_profile.json`. The frontend can be selected through
`PARTICLE_SLIDER_COMMAND`; it does not install or replace product dependencies.
`./train --config training.example.json --plan` inspects the request without
allocating a GPU. Execution and `--check` return the concrete readiness reason
before loading native model code.

Current readiness: **blocked**. Default source uses signed contrastive velocity MSE and has no shared lifecycle/full-state checkpoint bridge. The separate machine-local PG0.9.4 profile is explicit; this profile does not relabel default training as PG.

The example contains only the intended update budget and is not a claim that
this product can train. Existing commands, algorithms and dependency locks are
unchanged. A real state-preserving training factory must be integrated before
this profile can become executable; no random fixture or unrelated model is
substituted. Training, when supported, will not automatically search LRs,
render samples or publish artifacts.

## Existing local PG 0.9.4 training route

This workstation also has a separately verified **adapter-only** PG 0.9.4 route.
It uses the existing FP32 DiT/BF16 text-encoder cohort and does not change this
repository's default MSE trainer or dependencies:

```sh
./train --config character-appeal.local.json --profile training_profile.local-pg094.json
```

The later `--profile` explicitly selects the alternate profile. Add `--check`
for CPU/source validation. The profile is tied to this machine's runtime,
immutable source receipt, cached weights and shared GPU0 policy. It is not a
portable Modal profile, and does not add a particle-bank prior to the existing
adapter-only GAN. This command trains/exports only; it neither repeats the LR
search nor renders images. Resume accepts a complete existing checkpoint directory.
