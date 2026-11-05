# Current ParticleGAN learning rates

The public current MOSS configuration requires an explicit `lr`. The supplied
examples use G `0.0001`; without `d_lr_mult`, the official `bispf_optimized`
multiplier remains `1.5`, giving D `0.00015`. This change preserves those values.
The optimizer is ParticleGAN DualNorm, so historical Adam/AdamW values do not
establish an appropriate current rate. No current native tuned default is claimed.

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

Native calibration compares identical initialized state, data, stochastic streams, precision, declared schedule horizons and completed-update budgets on the actual pretrained model. Select and independently confirm rates using only held-out raw teacher/student relative SSE: cast student, teacher and neutral to FP64 before subtraction, pool positions within each example, divide by that example's teacher-edit energy, then average examples equally. Keep the teacher-only valid/excluded set fixed across candidates; zero-energy examples remain undefined, and student failures never remove examples. See the [common LR protocol](https://github.com/HyperGAN/particle-sliders/blob/main/packages/particle-sliders-core/LEARNING-RATE-CONTRACT.md).

Preservation/no-edit drift, late/weak-field scores and native media outputs are optional diagnostics; they cannot select, reject or delay a rate. After the native mechanical proof, candidate endpoint net displacement is diagnostic too. Retain exact source/model/data/numerical identity, actual optimizer-rate, finite native-update, checkpoint-continuation and raw-response replay checks. Tiny CPU software checks are not native LR evidence.
Save measured memory, time, all effective optimizer-group rates and the complete
runtime identity for each trial.

The machine-local native LR campaign uses a separately pinned core `a794e3b`,
FP32 DiT/BF16 text encoder and an immutable native overlay. Its measured rates
remain qualified only for that exact cohort. This public source uses core
`ddb4142` and the declared native dependencies. CPU optimizer/replay tests confirm
configuration and continuation, not transfer of native quality or convergence.
Reproduce the selected candidate through this canonical source before promoting
it as a public native recommendation. Existing historical configs, weights,
source locks and the local running overlay remain unchanged.
