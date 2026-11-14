# Memory requirements

| Operation / profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
|---|---:|---:|---:|---|
| Encode training prompts | 1 | 24 | 76 | ≤256 tokens;3 prompts; ≤256 tokens; B=1; BF16 DiT/TE; FP32 VAE/adapter/critic/activation envelope; no checkpointing |
| Train default512px | 1 | 32 | 76 | 512×512; ≤256 tokens; B=4; BF16 DiT/TE; FP32 VAE/adapter/critic/activation envelope; checkpointed |
| Render1024px all components GPU | 1 | 52 | 76 | 1024×1024; ≤4096 tokens; B=1; BF16 DiT/TE; FP32 VAE/adapter/critic/activation envelope; no checkpointing |
| Render1024px explicit offload | 1 | 28 | 76 | 1024×1024; ≤4096 tokens; B=1; BF16 DiT/TE; FP32 VAE/adapter/critic/activation envelope; no checkpointing |

These are **calculated planning budgets, not measured peaks or hard runtime ceilings**. GiB means 2³⁰ bytes. They describe the finite profiles above, one worker and one learned slider, with RAW weights and **no model-weight EMA**. A larger image, token sequence, image/reference count, logical batch, cache or number of simultaneous workers needs a new calculation; there is no finite universal maximum for unbounded inputs.

## Scope and residency

Base: **`OpenMOSS-Team/MOSS-Image-SFT@fc5de0d56fc910cbca6f6625a70f5f582b26deb1`**. Training excludes VAE. backend.encode_training_prompts starts with DiT on CPU, encodes on GPU, moves the encoder back to CPU, then moves DiT to GPU. The frozen encoder therefore remains host-resident during training. The renderer supports explicit model CPU offload; that mode is separately bounded, not silently assumed for the all-GPU row.

Default example512px/rank4/logical batch4/four rollout states/eight signed rows/256 tokens. The sampling row binds1024px,50 steps, CFG3 and4096 tokens (the inference default). No model-weight EMA or quantization is allocated. Optional offload is an existing product path, not a change made for this documentation.

The table's efficient-SDPA rows require a flash/efficient implementation. An automatic backend can fall back to dense math; use the dense row where provided, or add the dense term below before planning capacity. A planning row does not change code, precision, optimizer, rates, initialization, checkpoints or qualification.

## Frozen and trainable ledger

- BF16 DiT tensor bytes: **10,164,617,280 bytes** (9.467 GiB).
- BF16 encoder tensor bytes: **16,785,390,048 bytes** (15.633 GiB).
- FP32 VAE file upper bound: **249,521,372 bytes** (0.232 GiB).

DiT index10,164,617,280 bytes; encoder8,392,695,024 parameters /16,785,390,048 bytes. Model uses42 main +2 text-refinement blocks, width3072, FFN8192 and24 heads. The Qwen3.5 encoder has24 recurrent/linear-attention blocks and8 full-attention blocks: a dense36-layer autoregressive KV formula would be wrong.

Trainable allowance: **4,405,761 parameters**. 42×4×4×(3072+3072) +276,993 spatial-critic parameters = 4,405,761; no particle bank. FP32 trainable weights and gradients require `4A+4A` bytes. Frozen base parameters have no gradient or optimizer tensor allocation. Routed particle-bank parameters, where present, are counted once; a frozen response cache is not a trainable bank.

The current `bispf_optimized` PG0.9.5 recipe uses **DualNorm, momentum 0, direction_blend**; it is not assumed to be Adam. Persistent per-parameter moment tensors are0 in this profile; scalar counters are small. The peak ledger additionally reserves **`48A+32MiB` bytes** for protected normals, displacement/direction copies, FP64 promotion and thin-matrix SVD/Gram workspace. This is a conservative temporary-allocation allowance, not a claim that the optimizer uses no memory. Selecting historical `bispf`/Adam instead adds its actual state (typically another`8A` bytes for two FP32 moments); nonzero momentum or a different solver requires recalculation. Weight EMA adds0 bytes and must stay disabled.

## Numeric peak ledger (GiB before headroom)

| Profile | Frozen GPU weights | Activation / decode | Trainable + optimizer | Other GPU allowance | Host loading/cache subtotal |
|---|---:|---:|---:|---:|---:|
| Encode training prompts | 15.633 | 0.190 | 0.016 | 0.000 | 55.809 |
| Train default512px | 9.467 | 14.062 | 0.261 | 0.000 | 55.809 |
| Render1024px all components GPU | 25.332 | 12.000 | 0.016 | 0.000 | 55.809 |
| Render1024px explicit offload | 15.633 | 4.175 | 0.016 | 0.000 | 55.809 |

CPU rows place their active tensors in host RAM; their frozen/activation entries are not GPU residency. Sequential phase sub-ledgers are in the source-facts report.

Cache bound for the training profile: **8 rows**, each with three FP32 native fields, plus **3 cached prompts**. Tensor-only cache allowance: `12,582,912+6,291,456` bytes; masks/indices and serialization overhead remain covered by the explicit host subtotal and reserve. These bounds are inputs to this calculation, not new runtime limits.

## Activation, cache and host calculation

The auditable inputs and per-profile subtotals are in [source-facts.json](docs/memory/source-facts.json). For a transformer envelope, let `N=image tokens+bounded text tokens`, `D=hidden width`, `I=FFN width`, `H=attention heads`, `L=covered blocks`, and `B=simultaneously retained native graphs`:

- One-block inference allowance: `4B(16ND+6NI)` bytes. The4-byte coefficient deliberately covers FP32 intermediates even for BF16 weights;16 and6 allow Q/K/V/output, normalization/residual, gated FFN and adapter intermediates.
- Uncheckpointed training allowance: `4LB(16ND+6NI)`; checkpointed allowance: `4(4LBND+2B(16ND+6NI))`. Checkpointing saves layer inputs and recomputes block internals; it does not remove the frozen model, adapter or critic.
- Dense math fallback adds `8BH N²` bytes for inference and `16LBH N²` for uncheckpointed training, or `32BH N²` for the checkpointed current-block envelope. These are conservative materialized-score/probability allowances; flash/efficient SDPA avoids that quadratic storage. GQA expansion uses the query-head count for this bound.
- Decode allowance is separately based on bounded output/tile area and channel width; profile notes record the bound. Peak is the **maximum of sequential phases**, with the actual overlapping model weights added inside each phase. Sampling steps are sequential and do not multiply resident peak.
- Frozen CPU rows are bounded by the profile (three FP32 native fields per cached row, plus explicitly bounded prompt embeddings). Loader staging/serialization copies and the largest published shard are included in host subtotals. Host cache/page residency and kernel workspaces vary, so these are planning allowances, not an allocator theorem.

Final GPU budget is `ceil_to_4GiB(1.25×GPU subtotal+2GiB)`; host is `ceil_to_4GiB(1.25×host subtotal+4GiB)`. The explicit25% plus runtime reserve covers allocator fragmentation, CUDA context/libraries, kernel workspaces, small FP32 islands and Python/serialization overhead. These chosen reserve multipliers are explicit planning assumptions; architecture metadata alone cannot prove every allocator or fused-kernel peak. It is not borrowed from scheduler admission caps, card size or global free-memory floors. Do not add every row's peak together for sequential phases; parallel workers do add their simultaneous requirements.

This profile's text-state allowance is **53,477,376 bytes** (0.050 GiB): use_cache=False; conservative FP32 recurrent state scratch, not all-layer KV. It is already included in the phase or host allowance, not added twice. Text-only encoding has no generated-token loop. If the actual encoder returns KV state, it is explicitly included as `2×layers×B×KV_heads×tokens×head_dim×dtype_bytes`; cache-disabled encoders allocate no persistent autoregressive KV. Recurrent/linear-attention state is accounted separately where applicable. Arbitrary output-token generation and image-conditioned encoders are outside these profiles.

## Sources and reproducibility

- [src/moss_sliders/backend.py](src/moss_sliders/backend.py)
- [src/moss_sliders/vendor/pipeline_moss_image.py](src/moss_sliders/vendor/pipeline_moss_image.py)
- [src/moss_sliders/lora.py](src/moss_sliders/lora.py)
- [src/moss_sliders/gan.py](src/moss_sliders/gan.py)
- [src/moss_sliders/inference.py](src/moss_sliders/inference.py)
- [examples/warm-light.json](examples/warm-light.json)
- [Captured architecture, published file/index sizes and arithmetic](docs/memory/source-facts.json) retains exact model revisions, metadata SHA256s and direct Hugging Face config URLs. No weights were downloaded or opened for this calculation.
- Optimizer facts: the pinned ParticleGAN0.9.5 `recipes.py` (`bcap`/`bispf_optimized`, lines1435–1451), `optim/dualnorm.py` momentum/state branch, and `optim/direction_blend.py` FP64 normals. These are part of the dependency selected by the repository's existing core lock; their source facts are recorded in the combined calculation report.

Documentation was calculated from source and published metadata on 2026-11-14. No model construction, import of Torch/Diffusers, forward pass, benchmark, sampling, training or new GPU work was performed.
