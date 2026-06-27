# Reproducing Tables

`results/paper_reported/` contains values transcribed from the ECCV 2026 paper
and appendix. These files are provenance records, not local reproduction claims.

## Paper Environment

The paper reports:

- training in PyTorch on a single NVIDIA RTX 3090
- 100 epochs
- AdamW, learning rate `1e-4`, cosine decay
- BCE + Dice loss
- input resolution `256x256`
- ASGP `T=3`, `N=64`
- FPS at `256x256`, batch size 1, including DWT/IDWT, FA-Scan, ASGP, GFA, and BRM

## Promotion Rule

Move a result from `paper_reported` to `reproduced` only after recording:

- checkpoint checksum
- exact repository commit
- config
- dataset split
- seed
- GPU, CUDA, PyTorch, and dependency versions
- evaluation/profiling command used on the server
