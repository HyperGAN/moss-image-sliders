# Historical PG0.9.1 environment

These files are unchanged snapshots from product PR1 head
`160c701119e82da9c47600e98ce7fe6bf7f1c476` (use the exact Git revision recorded
in the run; the source commit is authoritative). They are not current install
instructions or a security/UI qualification. In particular the old full/UI locks
have incompatible dependencies and must not be used for the current package.

For a historical checkpoint, use a separate checkout/environment at its recorded
product commit and core commit. Do not install the current source against this
old lock or relabel checkpoint headers/metadata. The prior base path used
PG0.9.1 and core `cdfb80a76b882b4b73861b12f555263bd64ffeb7`. Older PG0.9.0,
signed-MSE pilots and source-bound active campaigns keep their own original
recorded recipes. Current code rejects unidentified or cross-cohort GAN packets
before attachment/state mutation; portable legacy v1 inference remains supported.

The concurrent `feat/unified-training-command-20261104` branch owns a separate
machine-local PG094 launcher bound to its recorded source/runtime. It is not
promoted to portable model qualification by this change and remains untouched.
