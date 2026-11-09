# Finished-output rate guidance: MOSS Character Appeal

Use the existing **G0.001 RAW32** Character Appeal checkpoint for this measured
native256 BF16/rank2/batch2/horizon128 cohort, at fixed **D0.018000000000000002**,
with no bank. The [portable record](finished-output-rate.json) preserves all four
primary candidate means and the selected per-case gains, including its negative
case. The primary mean is **0.0026179340978463492**.

The separate four-case fresh result is **−0.003141462802886963** and did not
support the primary gain. It uses seeds 103/104 on the same two neutral portraits;
the primary used 101/102. Keep the weaker fresh result visible and separate.
The primary best-tested choice remains G0.001; do not pool the scopes, rerank by
SSE or require another confirmation. Final Boss is a different control.

## Current selection method

Use one scalar:

```text
case_gain = intended_effect(ON) - intended_effect(exact matched OFF)
rate_score = arithmetic_mean(case_gain over the declared cases)
```

Use the highest existing score within the same concept, native profile, training
budget and horizon. Retain negative gains, native zeros and exact ties; technical
failures stay missing. Teacher/student SSE is diagnostic only. Preservation,
beauty, confidence and completion never select or break a tie. Preserve each
completed primary comparison and its separate fresh result; do not pool them.
No additional confirmation, repeated search or worse-arm completion is required.
Reuse the original RAW checkpoints, proofs, teachers, outputs and first scores.
A full training state resumes only its exact unfinished prefix; a RAW inference
export is not a training checkpoint. Do not change initialization or relabel an
old packet. Official scalar optimizer statistics stay unchanged.

## Apply the actual rate options

The existing [MOSS configuration and CLI](../src/moss_sliders/cli.py) accept this
fragment in the complete matching MOSS configuration:

```json
{"lr": 0.001, "d_lr_mult": 18.0}
```

`lr` is G and actual D is `lr * d_lr_mult`; retain the exact actual D above.
The existing `moss-sliders train`/`moss-hndl train` options are `--lr` and
`--d-lr-mult`. [The common training profile](../TRAINING.md) uses the same fields.
MOSS rejects `prior_lr_mult`: there is no particle bank. Its host also rejects an
`ema_decay` option, so **do not add that unsupported field**. This route already
uses RAW current weights without model/adapter/critic EMA; no averaging,
comparison or EMA selection is introduced. The portable record's
`common_rate_overrides.ema_decay=null` states the common policy, while its
`product_option_fragment` above contains only actual supported MOSS keys.

Keep the recorded model/revision, two neutral portraits, source, precision,
rank, batch, horizon and inference profile. This fragment is not a full config,
native admission or a portable copy of the private image consumer. The existing
public factory's qualification limits stay separate from the captured research
cohort. A G-only comparison holds absolute D fixed; changing G with a fixed
multiplier changes D too and is a coupled comparison. The retained
[G0.0001 RAW128 response-SSE preset](learning-rates.md#measured-native256-pg095-preset)
is historical diagnostic evidence, not the current finished-output choice.

## Availability and reproduction

The JSON record publishes rate options, scope, all selected per-case gains,
candidate means and hashes of the retained score records. It is a guidance
record, **not a complete training request or downloadable checkpoint**. Private
campaign paths are not portable links. The selected RAW, original conditioning,
full native configuration, frozen instrument and output bank are not bundled
here; the existing released adapter weights keep their separate provenance.
Thus the public files let a reader reproduce the arithmetic from the recorded
gains, but do not promise image regeneration from this repository alone.
For example, from the repository root:

```sh
python - <<'PYCODE'
import json
from math import fsum
from pathlib import Path
record = json.loads(Path("docs/finished-output-rate.json").read_text())
for name in ("primary", "separate_fresh_result"):
    gains = list(record[name]["per_case_gains"].values())
    print(name, "mean from gains:", fsum(gains) / len(gains),
          "original recorded mean:", record[name]["mean_gain"])
PYCODE
```

This checks the published arithmetic, with ordinary floating-point rounding;
it is not a new rating or a reproduction of the private instrument/images.

With access to the exact saved cohort, use its existing RAW and matched outputs;
do not repeat training or rating to use this result. For genuinely missing
measurements, freeze the intended-effect definition, instrument, cases and native
settings before the first observation, reuse matched OFFs, then retain every case
at equal weight. The score is model-instrument evidence, not human/statistical
validation, a universal optimum or a new default.
