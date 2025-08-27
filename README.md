# GUIAlignFusion: Progressive Gated Alignment-Fusion Network for GUI Retrieval

<p align="center">
  <img src="https://raw.githubusercontent.com/your-repo/GUIAlignFusion/main/assets/teaser.png" width="85%">
  <br>
  <em>Figure 1 – Two-stage pipeline: task-oriented CLIP fine-tuning + MFEDFR-Combiner fusion.</em>
</p>

GUIAlignFusion is an end-to-end **vision-language search engine** that retrieves GUI screenshots from natural-language edit instructions. It sets **new SOTA** on our 62 k-triplet GUI Layout DD dataset, surpassing the best baseline by **+2.34 pp Recall@50** and **+22 % human-rated usefulness**.

| Metric | GUIAlignFusion | Best Baseline | Δ |
|---|---|---|---|
| Recall@10 | **68.7 %** | 61.5 % | +7.2 pp |
| Recall@50 | **92.7 %** | 90.6 % | +2.1 pp |
| MRR | **0.374** | 0.333 | +12.3 % |

---

## 🚀 Quick Start

```bash
conda env create -f environment.yml && conda activate guiaf
python train.py --stage 1 --config configs/aggf.yml   # 12 min
python train.py --stage 2 --config configs/mfedfr.yml # 15 min
python eval.py --split test                           # 30 s
<p align="center">
  <img src="https://raw.githubusercontent.com/your-repo/GUIAlignFusion/main/assets/dataset.png" width="75%">
  <br>
  <em>Figure 2 – Automatic triplet generation pipeline.</em>
</p>
Method
Stage-1 AGGF (Attention-Guided Gated Fusion)
Progressive unfreezing after 30 % training
Bidirectional cross-attention replaces cosine similarity
Adaptive gating fuses image & text cues
<p align="center">
  <img src="https://raw.githubusercontent.com/your-repo/GUIAlignFusion/main/assets/stage1.png" width="70%">
</p>
Stage-2 MFEDFR-Combiner
Frozen CLIP, trainable combiner from scratch
Hierarchical cross-attention + 32-slot prototype cache
Dynamic gating + residual bypass for multiscale fusion
<p align="center">
  <img src="https://raw.githubusercontent.com/your-repo/GUIAlignFusion/main/assets/stage2.png" width="70%">
</p>
