# Paper-Code Alignment Audit

Audit scope:

- main paper PDF and LaTeX source
- appendix PDF
- legacy experiment tree
- public architecture tree: `FGOS-Net`

## Matched Items

| Claim | Paper / appendix | Old code path | Public code |
| --- | --- | --- | --- |
| Full model route | FGOS encoder + GFA/BRM | legacy full-hybrid builder | `build_fgosnet("eccv2026_paper")` |
| Encoder width/depth | `32-64-128-192`, `2-2-2-2` | full hybrid builder | `FGOSConfig` |
| ASGP | `T=3`, `N=64` | full hybrid builder | `ActiveSpectralGeometricProbing` |
| FA-Scan assignment | LL/HH Hilbert, LH horizontal, HL vertical | `freq_adaptive` route | `FrequencyAlignedScan.branch_modes` |
| LGB policy | EEGG best row | gate policy in full builder | stage index policy |
| Efficiency verification | server-side profiling | internal profiling scripts | coming soon |
| Runtime verification | server-side profiling | internal profiling output | coming soon |

## Open-Source Decisions

- Keep model architecture and paper-code mapping notes.
- Do not publish weights yet.
- Do not publish training, testing, benchmark, prediction, server sync, rebuttal,
  or baseline stub scripts in the first public tree.
- If the paper wording and old experiment code differ, treat the old full hybrid
  builder as the implementation source of truth.

## Server Verification Required

The local desktop environment is not the paper environment. Parameter, FLOPs,
FPS, and checkpoint claims should be verified on the server before publishing
weights or profiling records.
