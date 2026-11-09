# Current ParticleGAN learning rates

Use the existing [Character Appeal G0.001 RAW32 finished-output result](finished-output-learning-rates.md) and [portable values](finished-output-rate.json). The package defaults and old response-SSE records below are separate.

The public current MOSS configuration requires an explicit `lr`. The supplied
examples use G `0.0001`; without `d_lr_mult`, the official `bispf_optimized`
multiplier remains `1.5`, giving D `0.00015`. This change preserves those values.
The optimizer is ParticleGAN DualNorm, so historical Adam/AdamW values do not
establish an appropriate current rate. The measured preset below is optional;
existing example values and defaults remain unchanged.

## Measured native256 PG0.9.5 preset

**Historical response-SSE diagnostic, not the current finished-output selector.** All original G0.0001 RAW128 numbers, tables, source identities and reports below remain unchanged.

Measured on 2026-11-05, the selected native RAW-response recipe uses **G `0.0001`, D
`0.018000000000000002`, no particle bank**, with a common 128-update horizon.
The historical matching configuration used this fragment:

```json
{"lr": 0.0001, "d_lr_mult": 180.0}
```

The actual optimizer contained one G group and four D groups. All D groups had
that exact rate (`0x1.26e978d4fdf3cp-6`); the multiplier is passed to the shared
constructor, without patching an optimizer after construction. `prior_lr_mult`
remains unsupported.

This is a **source-qualified native256 research preset**: OpenMOSS-Team/MOSS-Image-SFT
revision `fc5de0d56fc910cbca6f6625a70f5f582b26deb1`, product `d2699d62`, core
`3dd672a7`/official PG0.9.5, and the existing HyperGAN/HNDL spatial backend.
The measured configuration used BF16 DiT and text encoder, FP32 rank-2/alpha-2
attention LoRA, native FP32 `[1,128,16,16]` velocity fields at 256×256 pixels,
batch 2, seed 17, maximum text length 256, and two frozen neutral Euler rollout
positions with both signed directions. It used `bispf_optimized`, RAW selection
and strength magnitude 1. The native API was `moss_sliders.native_hndl.build_hndl_run`; portable replay
used public `moss_sliders.artifacts.load_slider` and the shared
`particle_sliders.pg09_backends.MossResponseAdapter`.

| G | Actual D | RAW at 32 | RAW at 64 | RAW at 128 |
| --- | --- | --- | --- | --- |
| `0.00003` | `0.018000000000000002` | 1.0823743551 | — | — |
| **`0.0001`** | `0.018000000000000002` | 1.0469044241 | **0.9914087042** | **0.9907276965** |
| `0.0003` | `0.018000000000000002` | **1.0034826355** | 1.0181641004 | — |
| `0.001` | `0.018000000000000002` | 1.2362764365 | — | — |

All four finite candidates completed 32 updates under the already declared
128-update schedule. The top two completed matched 64-update budgets; the
interior winner then completed 128. Four development examples pool both rollout
positions within each scene/seed/sign example. Only the selected winner was
measured at 128, so this is the best tested recipe under that declared search,
not an all-rates-at-128 comparison or a universal optimum.

Independent sealed confirmation, first opened after the 128-update winner was
frozen, scored **0.9091853181345732** over **12 valid examples** (three new scenes,
two seeds and both signs; 24 response fields), with zero excluded examples.
The native reader constructed no game, critic, optimizer or training run and made
zero updates. Source-bound fresh teachers were fixed before the first adapted
student response. Public RAW/OFF replay, frozen-base identity and adapter
immutability passed. An independent two-thread CPU audit recomputed every saved numerator, teacher
energy, per-example ratio and oracle using the exact FP64 reduction; it matched
the reported mean and teacher-only membership exactly, without a model or CUDA.

The only selection/confirmation metric was RAW equal-example relative SSE, with
FP64 casts before subtraction. OFF, teacher and reversed-edit controls were
exactly 1, 0 and 4. Preservation, media quality and endpoint displacement were
not selection or rejection gates. Fresh two-update versus independent 1+1
continuation matched full state and metrics; those two duplicate mechanical
updates account for **258 physical completed calls / 256 logical updates**.

See the [sanitized source and result record](native-lr-confirmation-pg095-20261105.json).
No model weights, response tensors, checkpoints or paired corpus are distributed.
This measurement does not qualify production-resolution sampling, VAE decoding,
offload, UI installation or every frontend configuration. Changing model,
precision, geometry, rank, batch, conditioning or schedule requires a new cohort.

All current entry points use the same names and recipe fields:

| Surface | Generator/base rate | Critic multiplier |
| --- | --- | --- |
| `build_hndl_run` / `build_run` | required config `lr` | optional config `d_lr_mult` |
| Shared `particle-slider` MOSS profile | required JSON `lr` | optional JSON `d_lr_mult` |
| `moss-sliders train` / `moss-hndl train` | config or `--lr` | config or `--d-lr-mult` |
| `python -m moss_sliders.execution` | config or `--lr` | config or `--d-lr-mult` |

An explicit multiplier lets a generator-only search hold D constant: for G
`0.00001` and D `0.001`, use `lr=0.00001` and `d_lr_mult=100`. That pair is an
**experimental candidate**, not a selected default. All supplied rates must be
finite and positive. MOSS has no particle bank; `prior_lr_mult` is refused.

```json
{"lr": 0.00001, "d_lr_mult": 100}
```

These fields belong inside the complete existing MOSS config. CLI overrides take
precedence over that file. The shared core records resolved rates, optimizer
history, row/noise streams and source identities in full checkpoints. Resume
requires the same rates and source cohort; changing a multiplier starts a new
trial. No host learning-rate scheduler or host weight EMA is introduced.

## Current finished-output selection

Rank rates by one equal-case arithmetic mean of intended-effect intensity ON
minus exact matched OFF in finished outputs; higher wins. Retain negatives,
native zeros and exact ties. Technical failures remain missing evidence.
Teacher/student SSE is diagnostic only; preservation, beauty, confidence and
completion never rank or break ties. Reuse completed searches, saved RAWs,
proofs, teachers, outputs and first scores. Preserve separate fresh results
without pooling; no additional confirmation or worse-arm completion is required.
Keep each comparison's native profile, matched budget and original horizon.
Use RAW current weights with no model-weight EMA; scalar optimizer statistics
are unchanged. Follow the exact product-supported no-EMA configuration.

## Historical response-error protocol (diagnostic only)

Earlier cohorts selected and confirmed rates by held-out RAW teacher/student
relative SSE. Their archived reduction cast student, teacher and neutral to FP64
before subtraction, pooled positions within an example, divided by that example's
teacher-edit energy and averaged examples equally. Teacher-only valid/excluded
membership was fixed; zero-energy relative scores were undefined and student
failures did not remove examples. That historical protocol and all recorded
values retain their original scope; they are not the current finished-output
selector or a requirement for new confirmation. Native source/model/data,
finite-update, continuation and RAW replay evidence stays with each cohort;
reuse valid completed proofs rather than repeat them.


The historical response-error studies treated media and preservation as optional diagnostics. Current selection measures intended effect in finished images; no preservation/beauty/confidence/completion gate or extra confirmation is added.
Save measured memory, time, all effective optimizer-group rates and the complete
runtime identity for each trial.

Earlier machine-local campaigns used core `a794e3b` and immutable native overlays;
public PG094 used core `ddb4142`. Those historical observations retain their exact
source identity. The fresh PG0.9.5 result above was measured through the current
canonical source and new teachers/state; no historical packet was rebound to it.
Historical configs, weights, locks and receipts remain unchanged. The package
continues to pin core0.13/PG0.9.5 and retains its existing example defaults.
