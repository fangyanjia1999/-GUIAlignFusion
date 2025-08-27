# GUIAlignFusion: Progressive Gated Alignment-Fusion Network for GUI Retrieval
![Alt text](https://github.com/fangyanjia1999/-GUIAlignFusion/blob/main/Display/overall.PNG)

*Figure 1: Overview of our approach: including the creation of the datasets, vision-language model training, and development of the GUI search engine.*

## Introduction
In today's era of widespread mobile applications, GUI retrieval is a critical technology for AI-assisted design and development. It significantly boosts efficiency, impacting both user satisfaction and team productivity. However, traditional learning-based methods face bottlenecks, including high sensitivity to image quality, difficulties in achieving fine-grained visual-language alignment, and a reliance on precise keyword matching.

To address these issues, we propose **GUIAlignFusion**, a novel two-stage vision-language search engine for mobile GUI retrieval. Our method employs a progressive unfreezing strategy and an AGGF module to fine-tune the CLIP model with a ranking loss for joint visual-textual encoding to locate target GUIs. It introduces an MFEDFR-Combiner module that enhances cross-modal representations using attention mechanisms and residual gating, then dynamically fuses them into a unified vector mapped to target interfaces.

Comprehensive automated and human evaluations demonstrate that our method significantly outperforms baseline models, achieving 69% higher Recall@10, 93% higher Recall@50, and a 37% improvement in MRR.

## Method Overview
Our approach consists of two main stages:

### Stage 1: Feature Alignment Enhancement
![Alt text](https://github.com/fangyanjia1999/-GUIAlignFusion/blob/main/Display/stage1.png)
*Figure 2: In the first stage of training, we perform task-oriented fine-tuning of CLIP encoders to reduce the mismatch between large-scale pretraining and the downstream task.*

We freeze the CLIP encoders and only train the novel Attention Guided Gated Fusion (AGGF) module. After 30% of the training, we progressively unfreeze the vision encoder layers in an architecture-aware order to adapt the unified embedding space for downstream GUI retrieval.

### Stage 2: Feature Fusion Generation
![Alt text](https://github.com/fangyanjia1999/-GUIAlignFusion/blob/main/Display/stage2.jpg)
*Figure 3: In the second stage of training, we train from scratch a MFEDFR-Combiner network that learns to fuse the multimodal features extracted with CLIP encoders.*

![Alt text](https://github.com/fangyanjia1999/-GUIAlignFusion/blob/main/Display/Model2.png)
*Figure 4: MFEDFR-Combiner serves as the main network in the second stage for enhanced feature fusion. Starting from its baseline architecture (a), it is upgraded to version (b) by incorporating a Multiscale Fusion and Dynamic Fusion module.*

## Dataset Construction
We automatically construct a large-scale dataset, **GUI Layout Differences Description (GUI Layout DD)**, containing 62,530 triplets for composed GUI retrieval. It is constructed from RICO and RICO-Topic by leveraging computer vision techniques for component matching and GPT models for generating natural language edit instructions.

### Rico-Topic Dataset
We transformed the original RICO dataset into the Rico-Topic dataset by consolidating categories and applying automated prefiltering and manual annotation, ultimately curating 5,562 high-quality screenshots across 10 mutually exclusive themes.

### GUI Layout Differences Description Dataset
For triplet generation, we implemented an automated pipeline:
- GUI elements extracted via HSV conversion and contour detection using OpenCV
- Mechanical descriptions refined into natural language using GPT-2
- Component differences matched via Euclidean distance

This process produced structured triplets, each comprising a reference image, a difference description, and a target image.

## Implementation Details
We implement our CLIP-based cross-modal GUI retrieval system with a focus on three aspects: multistage training, dynamic optimization, and advanced fusion. Our setup is evaluated on a newly constructed dataset of 5,559 GUI triplets derived from RICO.

### Model Implementation
- AdamW optimizer with hierarchical learning rates (3e-5 for fusion modules, 3e-6 for CLIP parameters)
- Input GUI images resized to 288×288 pixels
- CLIP-encoded into 640-dimensional feature vectors
- Mixed precision computing and gradient clipping with max norm of 1.0
- Enhanced margin-based ranking loss with margin of 0.2

## Results
### Automated Evaluation
We conducted a systematic evaluation comparing our cross-modal GUI retrieval system against three baseline methods:
1. Vision-only CNN UI Autoencoder
2. Shallow fusion GUing model
3. Fixed fusion Combiner

![Alt text](https://github.com/fangyanjia1999/-GUIAlignFusion/blob/main/Display/login.png)
![Alt text](https://github.com/fangyanjia1999/-GUIAlignFusion/blob/main/Display/setting.png)
*Figure 5: Top-5 retrieved Target images on (a) Login; (b) Settings.*

![Alt text](https://github.com/fangyanjia1999/-GUIAlignFusion/blob/main/Display/relevance.jpg)  
![Alt text](https://github.com/fangyanjia1999/-GUIAlignFusion/blob/main/Display/diversity.jpg)
*Figure 6: Top-5 Visual Examples Demonstrating Retrieval on (a) relevance; (b) diversity.*

Our method consistently outperforms all baselines across all metrics:
![Alt text](https://github.com/fangyanjia1999/-GUIAlignFusion/blob/main/Display/T1.png)

![Alt text](https://github.com/fangyanjia1999/-GUIAlignFusion/blob/main/Display/T2.png)
### Ablation Study
We conducted a systematic ablation study on the three core modules of our MFEDFR-Combiner:

![Alt text](https://github.com/fangyanjia1999/-GUIAlignFusion/blob/main/Display/T3.png)
### User Evaluation
Blind assessments from six independent groups showed our method significantly outperformed all baseline models across three metrics:
- Relevance: 3.77 (37% improvement)
- Diversity: 4.10 (29% improvement)
- Usefulness: 3.55 (22% improvement)

![Alt text](https://github.com/fangyanjia1999/-GUIAlignFusion/blob/main/Display/T4.png)
Mann-Whitney U tests confirm statistically significant improvements across all metrics.

## Conclusion
This paper introduces GUIAlignFusion, a progressive fine-tuning strategy for GUI retrieval. The method begins by freezing the CLIP backbone to train only an attention-gated fusion module for rapid feature adaptation. It then progressively unfreezes vision encoder layers following an architecture-aware sequence, balancing stability with pretrained knowledge utilization. A dedicated Combiner network subsequently refines and fuses multimodal features, effectively bridging the domain gap between general pretraining and GUI-specific tasks while improving joint embedding additivity.

Evaluations on our self-constructed dataset demonstrate superior performance over strong baselines, with visualizations validating the efficacy of the tuning and fusion mechanisms.

## Acknowledgments
This work was supported by the 111 Center under Grant No. D23006, the National Foreign Expert Project of China under Grant No. D20240244, Dalian Major Projects of Basic Research under Grant No. 2023JJ11CG002, and the Interdisciplinary Research Project of Dalian University under Grant No. DLUXK-2025-QNLG-015.


## Contact
For questions about our work or code, please contact [Yanfang Jia](mailto:jiayanfang@dlu.edu.cn) or [Tianming Zhao](mailto:zhaotianming@dlu.edu.cn).

---

*This project is based on research conducted at Dalian University, School of Software Engineering.*
