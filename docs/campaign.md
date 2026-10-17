# Resumable private training campaigns

`tools/train_campaign.py` keeps the signed frozen-velocity MSE objective. It does
not implement an adversarial, particle, or BISPF training method.

The executed recipe fixes all source prompt pairs and seeds. Frozen targets cover
both neutral and positive paths from matched noise, ten MOSS Euler positions per
path, with both frozen prompt velocities evaluated at each state. The target is
`base + sign * (positive - neutral)`, for signs -1 and +1. Raw MSE is averaged over
both signs and eight independent states before one AdamW update. Rank/alpha are
four, learning rate 0.0003, clipping threshold one, AdamW defaults otherwise.

A separately seeded shuffled Cartesian state sampler replaces the coupled
row/timestep modulo counters. A full epoch covers every row, seed, path and time.
The sampler cursor/order, its RNG, CPU/CUDA RNG, adapter and complete optimizer
state are checkpointed. Only complete immutable checkpoint directories are
eligible for recovery; hashes and recipe identity are checked before restoring.
`training.pt` is a private, locally produced optimizer checkpoint, loaded with
`weights_only=True`. Portable inference exports remain adapter-only safetensors.

Every ten updates has a resumable checkpoint. All checkpoints are retained.
Updates 200/400/800/1200/1600 receive fixed held-out diagnostics and matched-seed
images alongside frozen Off and direct positive-prompt baselines. Baselines also
include CFG-one positive-prompt images. The primary image path uses official
MOSS 50-step CFG-three generation, with unconditional layer 22 skipped. Training
targets are unguided; this mismatch is documented, not silently changed.

Diagnostics report teacher/student edit RMS, cosine and relative error separately
for each sign on fixed held-out states from both trajectories at positions
0/3/6/9. `BEST-DIAGNOSTIC.json` identifies only the lowest development positive
edit MSE. It is not an automatic perceptual winner. All candidate images remain
available for visual comparison. No final-test characters select checkpoints.

The supervisor observes GPUs and kernel errors and stops only its own process
groups. Existing GPU services and the system cooling service remain untouched.
Workers pause at 76 C and resume at 70 C; 82 C is a hard stop. Memory reserves,
competition, live power limits and host memory are guarded. Signal stops occur at
safe cache-bundle or complete-optimizer boundaries. Inspect progress, checkpoint
metadata and supervisor receipts before resuming; never launch a duplicate worker.

The first twenty effective-batch-eight updates record measured active/wall speed.
The worker stops for review if the active training projection exceeds 16.5 hours
per concept, rather than silently accepting a material increase over the approved
rough estimate. Target preparation and image evaluation are reported separately.
Weights, optimizer state, targets and new examples remain private on NVMe.
