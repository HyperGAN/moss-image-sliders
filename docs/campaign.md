# Shared GAN campaign entrypoint

Use `moss-sliders train-campaign --cache CACHE --output NEW_OUTPUT` only after
coordinator admission. The command imports the installed
`particle_sliders.pg09_moss_campaign` provider and runner, which execute
`particle_sliders.pg09.ConditionalPG09`. It loads only the pinned transformer;
encoded captions and native targets come from the existing immutable cache.

`MossCampaignRows` validates recipe, prompt order, embeddings, target-index,
row/seed/split identity and target checksums. It uses a bounded CPU LRU. The two
signs share latent, native timestep and the cached positive-minus-neutral teacher.
The neutral caption is the teacher negative pole; the empty caption is the frozen
CFG-negative condition used for rendering. These roles are distinct.

Only the training split can calibrate response RMS or fit the GAN. The shared
critic uses native spatial fields and mask-weighted Qwen context/time/sign/strength
conditions. Generator and critic have separate balanced shuffled row streams and
reference noise streams. There is no local MSE objective, local optimizer, or new
particle prior. The default maximum budget is 3840 updates; selecting a stopping
checkpoint and native validation panels belongs to the campaign coordinator.

Full checkpoints contain the complete shared game and are CPU-mapped before
restore. A new output directory is required for continuation; `--resume` preserves
the original total budget. Source/runtime/config changes reject old packets.
Historical MSE packets and adapter-only exports cannot serve as GAN resumes.
Package relocation changes source identities. Current active native campaigns
remain on their original source and checkpoint lineage, untouched by this change.

The old MSE helper APIs fail before update. The old monitor and autonomous CUDA
smoke fail before querying GPUs, acquiring locks or spawning anything. Existing
active workers and resource guardians remain coordinator-owned.
