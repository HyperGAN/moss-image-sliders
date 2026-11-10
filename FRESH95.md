# Explicit cached fresh initialization

`moss_sliders.fresh95` is an opt-in library API for a newly declared MOSS cohort
with already completed native response data. It leaves the existing commands,
profiles, source guards, model, LoRA names, defaults and saved cohorts unchanged.
It requires the original core0.13.0 / ParticleGAN0.9.5 / HNDL0.10.0 runtime and
original MOSS native pipeline. It does not upgrade an older environment or make
an old FULL/RAW artifact compatible with a new profile.

The initializer acts on the same existing HNDL down Parameters. Their original
constructor uses `kaiming_uniform_(a=1)`: gain1 and mean square `1/fan_in`.
The public PG0.9.5 deterministic orthogonal initializer uses QR at that second
moment. It preserves Parameter and Module identity; it need not preserve the
old Uniform distribution's entrywise bound. UP stays zero/KEEP and the original
HNDL critic keeps its constructor values. There is no particle bank or router.
Pretrained model weights are never initialized by this policy.

## Supply completed native data and an owner

This API consumes original `PreparedResponse` and `ResponseRow` objects whose
neutral and target fields have already been computed. It does not run the
original teacher/trajectory collector or the legacy PG0.9.1 cached builder.

The completed cache must retain:

- Both signed rows, `+1` then `−1`, for every original rollout point, with the
  exact latent, raw timestep, strength, frozen token embeddings and masks.
- Full FP32 native `[1,C,H/16,W/16]` neutral/target fields. Shapes, signs and
  paired conditioning cannot be flattened, truncated or reconstructed.
- Original whole-output and Source file records for every completed row, plus
  the original frozen model ID/revision and whole-file identity records.

The original mask-weighted normalized neutral features, time, sign and strength
form the critic conditions; frozen edit RMS supplies the response scale. There
is no new conditioning formula or reconstruction loss.

The caller provides two explicit functions:

- `owner_admission(request)` authenticates the declared original base/output
  records and admits that operation's resources. It must return `True` only
  after those checks succeed.
- `component_factory(request)` supplies the original admitted
  `MOSSImagePipeline`, with its native transformer and required components on
  the declared device. The caller retains the pipeline's lifetime and device.

The library validates and binds those records, actual native classes, Source
guards and geometry. It does not independently prove which frozen weight bytes
the component factory loaded. It does not download models, admit GPUs or provide
a standalone admission CLI. Neither callback should collect replacement teachers
or reset an existing cohort.

```python
from pathlib import Path
from moss_sliders import MODEL_ID, MODEL_REVISION
from moss_sliders.fresh95 import save_prepared_cache

# The data/resource owner supplies these actual objects and authenticated
# records; these names are not loaders or synthetic stand-ins for native data.
config = dict(
    objective="pg09_gan_only", pg_version="0.9.5",
    rank=2, seed=7, resolution=256, rollout_steps=4, strength=1.0,
    batch_size=2, steps=128, lr=0.001, d_lr_mult=18.0, critic_width=32,
)
base_identity = dict(
    authenticated_original_identity, scope="pretrained-native",
    whole_files=authenticated_original_base_file_records,
)
assert base_identity["model"] == MODEL_ID
assert base_identity["revision"] == MODEL_REVISION
cache = save_prepared_cache(
    Path("outputs/new-cache.pt"), config=config, base_identity=base_identity,
    prepared=completed_original_signed_rows,
    teacher_origins=authenticated_completed_row_origins,
    cache_id="new-declared-completed-cache", owner_admission=owner_admission,
)
```

Each teacher-origin entry records `scope="completed-native"`,
`complete_prepared_output=True`, its original `signed_scale`, and nonempty
`whole_output_files` and `source_files` lists. A file record contains its
absolute `path`, positive `bytes` and whole-file `sha256`. These are truthful
original records authenticated by the owner, not newly invented receipts.
Preserve any additional original configuration/conditioning identity fields.
If the pipeline carries `_moss_training_context_identity`, the owner-supplied
`base_identity["conditioning"]` must match it exactly.
`save_prepared_cache` creates a new immutable cache and returns its typed
whole-file pin. It does not overwrite or migrate an old cache.

The example rates are explicit controls, not evidence that the new initializer
inherits a measured LR result. G0.001 with multiplier18 keeps absolute
D0.018000000000000002 and no bank. A new initializer profile is a new cohort.

## Fresh means one new initialization before optimizers

Choose `mode="fresh"`, a genuinely new cohort ID and a new directory. Supply no
resume packet. Intent, no-EMA policy, exact runtime and complete cache identity
are checked before the component factory or added components. The fresh receipt
is part of the immutable game signature before the shared core creates the
optimizers. Entry writes initialization metadata and FULL0/RAW0.

```python
from moss_sliders.fresh95 import open_training

with open_training(
    config, cache, Path("outputs/new-cohort"), mode="fresh",
    cohort_id="explicit-new-cohort", base_identity=base_identity,
    device="cuda:0", component_factory=component_factory,
    owner_admission=owner_admission, ema_decay=None,
) as session:
    # Each update and checkpoint is explicitly caller directed.
    metrics = session.update()
    full = session.save(Path("outputs/new-FULL1.pt"))
    raw = session.export(Path("outputs/new-RAW1.pt"))
```

Only fresh construction invokes the PG initializer. Training, evaluation,
exports and inference use raw current weights. `ema_decay=None` is required;
adapter, critic and bank weight averaging is unsupported. Upstream scalar
optimizer statistics remain unchanged and are not weight EMA.

## Resume the exact FULL without reinitializing it

Use the same configuration, cache, original identities and cohort ID, with
`mode="resume"`, its typed FULL pin and a new output directory. The initializer
is skipped. The original architecture is reconstructed and the entire typed
game state is strictly restored and compared, including both optimizers,
critic, sampling/exposure cursors, schedules and CPU/CUDA RNG. The total horizon
remains the original configured budget; opening the session performs no update.

```python
with open_training(
    config, cache, Path("outputs/resume-cohort"), mode="resume", resume=full,
    cohort_id="explicit-new-cohort", base_identity=base_identity,
    device="cuda:0", component_factory=component_factory,
    owner_admission=owner_admission, ema_decay=None,
) as restored:
    assert restored.inherited_steps == 1
```

Only FULL packets from this exact new profile can resume it. A RAW export,
legacy MSE packet, old campaign FULL or changed Source/configuration fails
closed. Use old saved artifacts under their original recorded runtime; never
reset or reinitialize them to adopt this API. Reuse completed LR searches and
credited updates rather than replaying them.

## Load RAW for native inference without a game

`open_inference` authenticates the typed RAW and original identity before
constructing components. It strictly loads the saved added-network tensors and
skips the completed cache payload, critic, game, optimizer and PG initializer.
`response` supplies the native velocity diagnostic; `render` calls the unchanged
MOSS prompt preflight, native encoder, scheduler, conditional-only CFG and VAE
route with the caller's fully staged components.

```python
from moss_sliders.fresh95 import open_inference

with open_inference(
    raw, component_factory=component_factory, owner_admission=owner_admission,
) as inference:
    image = inference.render(
        prompt, scale=1.0, seed=7, width=256, height=256,
        steps=50, guidance=3.0, output_type="pil",
    )
```

This is a callable route, not evidence that new-wheel full pretrained rendering
has run. The owner must stage native text/tokenizer/VAE components and admit the
full operation. Native prompt budgets fail before encoding instead of silently
truncating. No released weights or ordinary Generate readiness is claimed.

Close restores only this entry's hooks, per-forward scale checkpoints, original
flags and attribute presence. Once cleanup starts, update/inference is blocked.
Failed operations stay pending for retry; foreign replacements are retained and
reported. The borrowed pipeline is not closed, moved or unloaded by this entry.

## What has been checked

Four new installed software cases passed with one total tiny random-model
update. They exercised receipt-before-optimizer ordering, original Parameter
identity, zero-UP/critic KEEP, analytic QR moments and caller RNG, strict whole
FULL/RAW skips, original runtime guards and retryable foreign-safe cleanup.
Synthetic completed rows replaced only data collection; no real teacher,
pretrained model, native full render, old proof or LR search ran.

That installation used an explicitly owned setuptools software candidate with
genuine generated distribution RECORDs and original core/PG/HNDL guards. The
repository's original hatchling1.28 build was not exercised by those cases.
Original hatchling/public packaging and full pretrained/public native execution
remain unqualified. The new namespace falls under the existing
`src/moss_sliders` wheel package selection; no metadata, dependency, version,
entrypoint or default is changed by adding these files.

The existing Character Appeal primary G0.001 RAW32 result remains
`+0.0026179340978463492`; its separate fresh result remains
`−0.003141462802886963`. They retain their original cohorts and are not pooled,
rerated or attributed to this new initializer. See the existing
[finished-output guidance](docs/finished-output-learning-rates.md). New intended
effect measurements use raw finished outputs and equal matched-case ON−OFF;
teacher/student SSE remains diagnostic only.
