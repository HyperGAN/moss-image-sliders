# Verified MOSS integration

Verified 2026-10-17 against model commit
[`fc5de0d56fc910cbca6f6625a70f5f582b26deb1`](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/tree/fc5de0d56fc910cbca6f6625a70f5f582b26deb1).
No weights are vendored or redistributed.

| Component | Official configuration / integration |
| --- | --- |
| Transformer | MOSSImageTransformer, 42 single-stream DiT blocks, dimension 3072, 24 heads, FFN 8192, text dimension 4096 |
| Latents | [B,128,H/16,W/16], FP32 initial Gaussian noise; not FLUX packed token inputs |
| Attention | `layers.N.proj.q`, `.k`, `.v`, `.o`; these Linear modules are the only adapter targets |
| Text encoder | Qwen3_5Model; frozen; official thinking-enabled chat template; last_hidden_state plus attention mask |
| VAE | AutoencoderKLFlux2, 32 channels then 2×2 patching to 128; official BN normalization/denormalization |
| Scheduler | MOSSImageScheduler, flow Euler, shift 5, nearest training-grid schedule; transformer timestep in scheduler units |
| Precision | BF16 DiT/encoder, FP32 VAE, noise and CFG composition |
| Images | Positive width/height multiples of 16; official recommendation ≥50 sampling steps |
| Sampling guidance | CFG 3, empty negative prompt, negative branch skips layer 22 by default |
| License | Weights and bundled inference source Apache-2.0, full license retained |

Sources: [model card](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT),
[model index](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/fc5de0d56fc910cbca6f6625a70f5f582b26deb1/model_index.json),
[transformer config](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/fc5de0d56fc910cbca6f6625a70f5f582b26deb1/transformer/config.json),
[pipeline source](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/fc5de0d56fc910cbca6f6625a70f5f582b26deb1/pipeline_moss_image.py),
[official user guide](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md).

The advertised quick-start loads `DiffusionPipeline.from_pretrained` with
`custom_pipeline="pipeline_moss_image"`, `trust_remote_code=True`, and dtype mapping
`{"default": torch.bfloat16, "vae": torch.float32}`. We instead load the same pinned
components using the unmodified local copy of those classes. AutoModel and AutoTokenizer
use library-provided implementations; arbitrary remote Python is not required.
The exact vendored source SHA256 is tested in CI.

The official transformer does not inherit Diffusers' PEFT adapter mixin, and the
pipeline does not expose `load_lora_weights` or `set_adapters`. Calling the generic
FLUX/Qwen LoRA APIs would therefore be incorrect. We attach removable forward hooks
to the real main-block Linear projections, preserving `.weight` access in the MOSS
forward. Adapters store only float32 down/up matrices with an exact target list and
base-model revision. This format is project-specific, not claimed compatible with
ComfyUI/PEFT adapter loaders. Rank/alpha follow particle-sliders' alpha/rank scaling.

The official guide recommends freezing the text encoder during post-training. We
also freeze the VAE (unused in training) and all transformer base weights. Training
uses the official scheduler rather than assuming a DDPM epsilon loss. This new
contrastive velocity recipe is experimental; the official release does not provide
a slider training objective or quality guarantee.
