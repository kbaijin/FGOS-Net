# Method to Code Map

| Paper Component | Old experiment code | Public code | Status |
| --- | --- | --- | --- |
| Full FGOS-Net builder | legacy full-hybrid builder | `fgos.build_fgosnet("eccv2026_paper")` | aligned to the full hybrid route |
| FGOS Encoder | `models/network_fgos.py`, `models/fgos/fgos_block.py` | `fgos/models/fgosnet.py`, `fgos/models/blocks.py` | public architecture |
| DWT/IDWT | `models/fgos/dwt_modules.py` | `fgos/models/wavelet.py` | public PyTorch Haar implementation |
| HF-conditioned Align | `FGOS_Block.offset_predictor` | `FGOSBlock.offset_predictor` | public implementation |
| FA-Scan | `ssm_scan_mode="freq_adaptive"` | `fgos/models/scan.py` | LL/HH Hilbert, LH horizontal, HL vertical |
| LGB | `light_gate_bottleneck.py` | `fgos/models/lgb.py` | stage-adaptive ECA/GSE |
| ASGP | `models/fgos/asgp.py` | `fgos/models/asgp.py` | `N=64`, `T=3`; paper/fast modes |
| Hybrid GFA decoder | `SegLineFGOS_Hybrid.decoder` | `HybridGFADecoder` | public architecture |
| BRM head | `MultiBranchFusionHead` | `MultiBranchBRMHead` | public architecture |

The public tree intentionally excludes training/evaluation scripts, checkpoints,
private paths, baseline stubs, and rebuttal artifacts.
