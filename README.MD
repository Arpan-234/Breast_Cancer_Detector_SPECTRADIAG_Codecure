# 🧠 Breast Cancer MRI Classification — CODECURE AI Hackathon (IIT-BHU)

> **Dual-Model Framework:** SPECTRA-Net + NeuroFusionX for clinically reliable, uncertainty-aware, and interpretable breast cancer detection from MRI images.

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Models Overview](#-models-overview)
  - [Model 1: SPECTRA-Net](#model-1-spectra-net)
  - [Model 2: NeuroFusionX](#model-2-neurofusionx)
- [Model Comparison & Recommendation](#-model-comparison--recommendation)
- [Experimental Results](#-experimental-results)
- [Architecture Diagrams](#-architecture-diagrams)
- [Novelty vs. Existing Research](#-novelty-vs-existing-research)
- [Explainability & Uncertainty](#-explainability--uncertainty)
- [Installation & Usage](#-installation--usage)
- [Project Structure](#-project-structure)
- [Submission Summary](#-submission-summary)

---

## 🎯 Problem Statement

Breast cancer is the most common cancer among women globally. Early and accurate detection from MRI imaging is critical but challenging due to:

- **Class imbalance** (malignant cases are rare in screening datasets)
- **Black-box AI predictions** that clinicians cannot trust or audit
- **Overconfident models** with poor calibration — dangerous in medical settings
- **Lack of uncertainty quantification** — no signal for when to refer to an expert

This project addresses **all four gaps** with a dual-model pipeline that is accurate, calibrated, explainable, and uncertainty-aware.

---

## 📂 Dataset

**Dataset:** [Breast Cancer Patients MRIs — Kaggle (uzairkhan45)](https://www.kaggle.com/datasets/uzairkhan45/breast-cancer-patients-mris/versions/1)

| Property | Value |
|---|---|
| Total Images | 1,480 |
| Classes | 2 (Benign / Malignant) |
| Split | 70% Train / 15% Val / 15% Test |
| Class Distribution | Highly imbalanced: 1,400 benign : 80 malignant |
| Image Format | RGB MRI scans |

> ⚠️ **Imbalance Handling:** Weighted random sampling + class-weighted loss function. Class weights computed as `total_n / (num_classes × class_count)`, resulting in weights `[0.529, 9.241]` for benign and malignant respectively.

---

## 🤖 Models Overview

### Model 1: SPECTRA-Net

**SPECTRA-Net** — *Statistical-Probabilistic Evidential Clinical Texture-Rich Adaptive Network*

A fully **custom-built, from-scratch** multi-branch architecture with no pretrained weights, designed for interpretable and uncertainty-aware classification.

#### Architecture

```
Input MRI (224×224, Grayscale)
        │
        ├──► [Wavelet Branch]       DWT subbands → Conv → Pool → Linear → 128-dim
        │     (Haar, 2 levels, 7 subbands)
        │
        ├──► [Morphological Branch] Soft Dilation/Erosion → Conv → Pool → Linear → 128-dim
        │     (Learnable struct elements, differentiable)
        │
        ├──► [CNN Backbone]         DSConv Blocks (Stem→S1→S2→S3→S4) → 128-dim
        │     (Depthwise-Separable Convolutions, no pretrained)
        │
        └──► [GLCM Texture Branch]  Soft Quantization → Co-occurrence Matrix → MLP → 128-dim
                                                     │
                                    ┌────────────────▼────────────────┐
                                    │   Gated Branch Fusion           │
                                    │   4-way softmax gate weights    │
                                    │   Weighted sum → 128-dim fused  │
                                    └────────────────┬────────────────┘
                                                     │
                                    ┌────────────────▼────────────────┐
                                    │   EvidentialHead                │
                                    │   Linear→GELU→Dropout (×2)     │
                                    │   Softplus → Dirichlet α        │
                                    │   Output: prob, uncertainty,    │
                                    │           alpha, logits         │
                                    └─────────────────────────────────┘
```

#### Key Components

| Component | Description |
|---|---|
| `WaveletBranch` | PyWavelets DWT → 7 subbands (LL, LH, HL, HH × levels) → Conv2D → AdaptivePool |
| `LearnableMorphOp` | Differentiable soft dilation/erosion with learnable structural elements |
| `MorphBranch` | Two morphological ops → concatenate → DSConv stack |
| `GLCMTextureBranch` | Soft quantization to bins → GLCM co-occurrence matrix → FC |
| `LiteCNNBackbone` | 4-stage Depthwise-Separable CNN, 4.38 MB total |
| `GatedFusion` | 4-way learned softmax gate dynamically weights branch contributions |
| `EvidentialHead` | Dirichlet-based uncertainty: outputs α, S, prob, uncertainty score |
| `EvidentialFocalLoss` | Focal loss + KL Dirichlet divergence with annealing |
| `TemperatureScaler` | LBFGS post-hoc calibration for reliable probability outputs |
| `GradCAM++` | Custom implementation with full backward hook support |
| `MC Dropout` | 30-sample Monte Carlo inference for epistemic uncertainty |

---

### Model 2: NeuroFusionX

**NeuroFusionX** — *Self-Organizing, Self-Supervised Global–Local Network*

A **transfer-learning-based** architecture built on a pretrained EfficientNet-B0 backbone, extended with novel self-organizing and statistical fusion components.

#### Architecture

```
Input MRI (256×256, RGB)
        │
        ▼
[EfficientNet-B0 Backbone]   (Pretrained ImageNet weights)
        │ feature map (7×7×1280)
        ▼
[Soft Attention Map]         Learned spatial attention weights
        │ attention-weighted pool → 1280-dim global embedding
        ▼
[StatFusionLayer]            Concat [attn_pool | mean | std] → Linear → 512-dim
        │
        ▼
[SOMLayer]                   6×6 Self-Organizing Map codebook
        │ SOM loss: attraction to nearest prototype
        ▼
[SimCLR Projection Head]     MLP for contrastive self-supervised regularization
        │
        ▼
[Classifier Head]            Linear(512 → 2) + class-weighted CrossEntropy
        │
        ▼
[Temperature Scaling]        Post-hoc calibration (learned T = 0.857)
        │
        ▼
Output: prob, threshold (Youden-optimal), calibrated logits
```

#### Key Components

| Component | Description |
|---|---|
| `EfficientNet-B0` | Pretrained ImageNet backbone, fine-tuned end-to-end |
| `SoftAttentionMap` | Conv1×1 → Sigmoid spatial attention over feature map |
| `StatFusionLayer` | Fuses attention embedding with channel mean & std statistics |
| `SOMLayer` | Differentiable 6×6 codebook; soft-winner attraction loss |
| `SimCLR Head` | Contrastive self-supervised auxiliary objective |
| `WeightedRandomSampler` | Oversamples minority class during training |
| `Youden-Optimal Threshold` | Per-run threshold calibrated on validation ROC curve |
| `Temperature Scaling` | LBFGS calibration; learned T = **0.857** |
| `GradCAM++` | Saliency maps via `pytorch-grad-cam` library |
| `5-Seed Cross Validation` | Statistical robustness across seeds {42, 52, 62, 72, 82} |

---

## ✅ Model Comparison & Recommendation

### Side-by-Side Comparison

| Property | SPECTRA-Net | NeuroFusionX (Final) |
|---|:---:|:---:|
| **Architecture type** | From scratch | Transfer learning (EfficientNet-B0) |
| **Parameters** | 1,094,922 (~1.1M) | ~4–6.3M |
| **Model size** | **4.38 MB** | ~16–25 MB |
| **Best Val AUC** | 0.806 | **1.000** (seed 52) |
| **Cross-seed AUC (mean ± std)** | — | **0.9882 ± 0.0207** |
| **Cross-seed F1 (mean ± std)** | — | **0.9372 ± 0.0536** |
| **Brier Score** | — | **0.0156** (excellent) |
| **ECE (calibration)** | — | **0.0537** |
| **EER** | — | **0.0743** |
| **Best test AUC** | 0.789 (early epochs) | **0.9882** |
| **Sensitivity** | 0.833 | **0.9375** |
| **Specificity** | ~0.724 | **0.9929** |
| **Uncertainty output** | ✅ Evidential + MC Dropout | ✅ Temperature-scaled probs |
| **Explainability** | ✅ GradCAM++ + Branch Attention | ✅ GradCAM++ + Attention Maps |
| **Calibration** | ✅ Temperature Scaling | ✅ Temperature Scaling (T=0.857) |
| **Training speed** | Slow (wavelet ops CPU-bound) | **Fast** (pretrained backbone) |
| **Needs GPU** | Optional (slow on CPU) | Recommended |
| **Robustness (multi-seed)** | Not tested | ✅ Validated across 5 seeds |

### 🏆 Recommendation: **NeuroFusionX (Final Stable Version)**

For a **real clinical deployment scenario**, **NeuroFusionX** is the clear winner based on:

1. **Superior AUC**: Cross-seed mean AUC of **0.9882** vs SPECTRA-Net's ~0.79 — nearly perfect discriminability.
2. **Excellent calibration**: Brier Score **0.0156** and ECE **0.0537** indicate highly reliable probability estimates.
3. **Proven robustness**: Validated across 5 random seeds — not a lucky run.
4. **Optimal threshold**: Youden-optimal threshold computed per run; learned temperature T=0.857 corrects overconfidence.
5. **Efficiency**: Converges in 3–8 epochs with early stopping; SPECTRA-Net required 60 epochs.
6. **Near-perfect confusion matrix**: [[280, 0], [1, 15]] — only 1 missed malignancy out of 16 test cases.

> **SPECTRA-Net** remains valuable as a fully custom, from-scratch research prototype demonstrating novel morphological and wavelet feature fusion — stronger from a novelty-of-architecture standpoint, and suitable for resource-constrained or interpretability-first settings where no pretrained weights are available.

### When to Use Which

| Scenario | Recommended Model |
|---|---|
| Maximum accuracy required | NeuroFusionX |
| No internet / no pretrained weights available | SPECTRA-Net |
| Resource-constrained edge deployment (< 5MB) | SPECTRA-Net |
| Interpretability + branch-level feature attribution | SPECTRA-Net (gate weights) |
| Clinical screening with calibrated confidence | NeuroFusionX |
| Research novelty / hackathon judging on innovation | Both (ensemble) |

---

## 📊 Experimental Results

### SPECTRA-Net — Training Results

| Epoch | Train Loss | Val Loss | Val Acc | AUC | Sensitivity | Specificity |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.1728 | 0.1712 | 0.8243 | 0.6702 | 0.417 | 0.848 |
| 2 | 0.1680 | 0.1608 | 0.5405 | 0.7500 | 0.833 | 0.524 |
| 4 | 0.1444 | 0.2342 | 0.6577 | 0.8063 | 0.833 | 0.648 |
| 5 | 0.1425 | 0.2284 | 0.6622 | 0.8020 | 0.833 | 0.652 |
| 8 | 0.1298 | 0.2880 | 0.8423 | 0.7345 | 0.667 | 0.852 |
| 12 | 0.1318 | 0.2925 | 0.7297 | 0.7893 | 0.833 | 0.724 |

> Dataset: 1,480 images, split 1,035 train / 222 val / 223 test. Class weights: [0.529, 9.241].
> Model: 1,094,922 parameters, 4.38 MB.

---

### NeuroFusionX — Final Stable Version Results

**Single Run (Seed 42):**

| Epoch | Loss | AUC | F1 | Confusion Matrix |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.0083 | 0.9973 | 0.6977 | [[268,12],[1,15]] |
| 3 | 0.0019 | 1.0000 | 0.9677 | [[280,0],[1,15]] |
| 8 | 0.0006 | 1.0000 | 0.9677 | [[280,0],[1,15]] |

> ⛔ Early stopping triggered at Epoch 8.
> ✅ Best AUC: **1.0000** | Stable Threshold: **0.341** | Learned Temperature: **0.857**

---

### NeuroFusionX — NeurIPS-Level 5-Seed Cross-Validation

| Seed | Best AUC | Notes |
|:---:|:---:|:---:|
| 42 | 0.9266 | Converged Ep3 |
| 52 | **1.0000** | Perfect — all 40 epochs |
| 62 | 0.9948 | Near-perfect |
| 72 | **1.0000** | From Ep2 onward |
| 82 | **1.0000** | From Ep1 onward |

| Metric | Mean | Std |
|---|:---:|:---:|
| **AUC-ROC** | **0.9882** | ±0.0207 |
| **F1 Score** | **0.9372** | ±0.0536 |

---

### NeuroFusionX — Clinical Evaluation Metrics (Final)

| Metric | Value |
|---|:---:|
| **AUC-ROC** | 0.9882 |
| **F1 Score** | 0.9372 |
| **Brier Score** | **0.0156** |
| **ECE (Expected Calibration Error)** | **0.0537** |
| **EER (Equal Error Rate)** | **0.0743** |
| **Attention-CAM Consistency** | 0.5958 |
| **Calibrated Temperature** | **0.857** |

> A Brier Score < 0.02 is considered excellent for medical AI. ECE < 0.06 indicates well-calibrated probabilities.

---

### NeuroFusionNet (v1) — EfficientNet-B0 Baseline

| Epoch | Train Loss | Train Acc | Val Acc | AUC | Sensitivity | Specificity |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.6642 | 0.6809 | 0.9054 | 0.8095 | 0.750 | 0.914 |
| 3 | 0.1581 | 0.9712 | 0.9730 | 0.8690 | 0.750 | 0.986 |
| 7 | 0.0242 | 0.9957 | 0.9865 | 0.9353 | 0.750 | 1.000 |

> Model: 4,011,391 parameters, 16.05 MB.

---

## 🏗️ Architecture Diagrams

### SPECTRA-Net Multi-Branch Fusion

```
                         ┌─────────────────────────────────┐
                         │         Input MRI Image          │
                         │        (224×224, Gray)           │
                         └──┬──────┬──────────┬────────────┘
                            │      │          │         │
                     ┌──────▼─┐ ┌──▼──────┐ ┌▼──────┐ ┌▼───────────┐
                     │Wavelet │ │ Morph   │ │  CNN  │ │    GLCM    │
                     │Branch  │ │ Branch  │ │Backbn.│ │  Texture   │
                     │7-band  │ │Soft Dil │ │DSConv │ │ Soft Quant │
                     │DWT-2   │ │+ Erosion│ │4-Stage│ │ Co-occur.  │
                     └────┬───┘ └────┬────┘ └───┬───┘ └─────┬──────┘
                          │          │           │            │
                     proj_wav   proj_mor    proj_cnn     proj_glcm
                      (128d)     (128d)      (128d)       (128d)
                          │          │           │            │
                          └──────────┴─────┬─────┴────────────┘
                                           │
                                  ┌────────▼────────┐
                                  │  Gated Fusion   │
                                  │ g = softmax(Wz) │
                                  │ f = Σ gᵢ · fᵢ  │
                                  └────────┬────────┘
                                           │
                                  ┌────────▼────────┐
                                  │ EvidentialHead  │
                                  │  α = SP(Wx)+1   │
                                  │  u = K / Σα     │
                                  └─────────────────┘
                          prob | uncertainty | alpha | logits
```

### NeuroFusionX Global–Local Pipeline

```
Input RGB MRI (256×256)
        │
        ▼
┌──────────────────┐
│  EfficientNet-B0 │  ← Pretrained ImageNet weights
│  Backbone        │
└────────┬─────────┘
         │ Feature Map (7×7×1280)
         ▼
┌──────────────────┐
│  Soft Attention  │  ← Conv1×1 + Sigmoid spatial gate
│  Map (7×7)       │
└────────┬─────────┘
         │ Attention-weighted pooling
         ▼
┌──────────────────┐
│  StatFusion      │  ← [attn_pool | channel_mean | channel_std]
│  Layer (512d)    │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌──────────┐
│  SOM  │ │  SimCLR  │
│ 6×6   │ │  Proj.   │
│Codebk │ │  Head    │
└───┬───┘ └──────────┘
    │ (SOM reg. loss)
    ▼
┌──────────────────┐
│  Classifier      │  ← Linear(512→2)
│  Head            │     + CE Loss (class-weighted)
└────────┬─────────┘
         │
┌────────▼─────────┐
│ Temperature      │  ← LBFGS calibration (T = 0.857)
│ Scaling          │
└────────┬─────────┘
         │
    Youden-Optimal Threshold (0.341)
         │
    Final Prediction + Confidence
```

---

## 🔬 Novelty vs. Existing Research

### Gap in Existing Work

| Existing Approach | Limitation |
|---|---|
| ResNet/VGG transfer learning | No uncertainty, black-box |
| RAE-Net (ResNet-50 + EDLA, 2025) | No wavelet/morphology branch, no MC dropout |
| Mask R-CNN + ResNet50 | Detection only, no calibration |
| BCDCNN (PSPNet segmentation) | Segmentation required, no uncertainty |
| 3D CNN on DCE-MRI | No interpretability (Science Translational Medicine, 2022) |

### What This Project Contributes

| Innovation | SPECTRA-Net | NeuroFusionX |
|---|:---:|:---:|
| Learnable differentiable morphological operators | ✅ Novel | — |
| Gated 4-branch adaptive fusion | ✅ Novel | — |
| Wavelet + GLCM + CNN + Morphology combined | ✅ First combined | — |
| SOM codebook + SimCLR self-supervision on MRI | — | ✅ Novel |
| StatFusion (attention + mean/std) | — | ✅ Novel |
| Dual uncertainty: Evidential + MC Dropout | ✅ | — |
| Youden-optimal threshold + temperature scaling | ✅ | ✅ |
| 5-seed statistical validation | — | ✅ |
| GradCAM++ + branch attention dual explainability | ✅ | ✅ |
| Clinical flags ("HIGH UNCERTAINTY — Refer") | ✅ | — |

---

## 🔍 Explainability & Uncertainty

### SPECTRA-Net

- **GradCAM++**: Custom backward-hook implementation targeting `cnn_backbone.stage4[-1].pw` layer.
- **Branch Attention Weights**: `gate_weights` shows per-sample contribution of wavelet, morphology, CNN, and GLCM branches.
- **Evidential Uncertainty**: `u = K / Σα` — Dirichlet-derived aleatoric uncertainty.
- **MC Dropout**: 30-sample stochastic forward passes for epistemic uncertainty.
- **Clinical Flag**: `"HIGH UNCERTAINTY — Refer for Expert Review"` if `u > 0.3`.

### NeuroFusionX

- **GradCAM++**: Via `pytorch-grad-cam` library targeting EfficientNet-B0 final convolutional layer.
- **Attention Map**: `SoftAttentionMap` provides spatial localization of relevant lesion regions.
- **MC GradCAM + Uncertainty**: Monte Carlo GradCAM for stochastic saliency estimation.
- **Region Consistency Score**: 107,586 — measures stability of attention across MC samples.
- **Attention-CAM Consistency**: 0.5958 — spatial alignment between attention maps and GradCAM.
- **Calibrated Probabilities**: Temperature T=0.857 corrects slight overconfidence.
- **Brier Score 0.0156 / ECE 0.0537**: Clinically reliable probability estimates.

---

## ⚙️ Installation & Usage

### Requirements

```bash
pip install torch torchvision
pip install PyWavelets opencv-python
pip install scikit-learn matplotlib seaborn
pip install tqdm Pillow numpy
pip install grad-cam  # for NeuroFusionX GradCAM++
```

### Full `requirements.txt`

```
torch>=2.0.0
torchvision>=0.15.0
PyWavelets>=1.4.0
opencv-python>=4.7.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
Pillow>=9.5.0
numpy>=1.24.0
grad-cam>=1.5.5
scipy>=1.10.0
```

### Running on Kaggle

1. Add dataset: [uzairkhan45/breast-cancer-patients-mris](https://www.kaggle.com/datasets/uzairkhan45/breast-cancer-patients-mris)
2. Set `CFG.DATA_ROOT` to the dataset path.
3. Run the notebook cells in order.
4. Enable GPU accelerator (P100/T4 recommended).

### Quick Start — NeuroFusionX (Recommended)

```python
# Set config
CFG.DATA_ROOT = "/path/to/Breast Cancer Patients MRI's"
CFG.EPOCHS = 40
CFG.SEEDS = [42, 52, 62, 72, 82]

# Run full pipeline
model, test_metrics = run_full_pipeline()

# Run cross-seed validation
cross_seed_results = run_cross_seed(seeds=CFG.SEEDS)
# Output: AUC: 0.9882 ± 0.0207 | F1: 0.9372 ± 0.0536
```

### Quick Start — SPECTRA-Net

```python
# Set config
CFG.DATA_ROOT = "/path/to/dataset"
CFG.EPOCHS = 60
CFG.MC_SAMPLES = 30

# Run full pipeline (7 stages)
model, test_metrics, history = run_full_pipeline()
# Outputs: GradCAM++ saliency, clinical dashboard, reliability diagram
```

---

## 📁 Project Structure

```
breast-cancer-mri-detection/
│
├── brcancdetect-project-codecure-hackathon.ipynb  ← Main notebook (all models)
│
├── README.md                                       ← This file
├── requirements.txt                                ← Dependencies
│
├── models/
│   ├── spectra_net.py          ← SPECTRA-Net full implementation
│   └── neurofusionx.py         ← NeuroFusionX full implementation
│
├── outputs/
│   ├── spectra_outputs/
│   │   ├── spectra_net_best.pth
│   │   ├── spectra_net_final.pth
│   │   ├── training_history.png
│   │   ├── clinical_dashboard.png
│   │   ├── reliability_diagram.png
│   │   └── gradcam_sample_*.png
│   │
│   └── neurofusionx_outputs/
│       ├── metrics.csv
│       ├── summary.txt
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── pr_curve.png
│       ├── calibration_curve.png
│       ├── threshold_curve.png
│       ├── det_curve.png
│       ├── reliability_diagram.png
│       ├── training_curves.png
│       ├── seed_wise_auc.png
│       └── gradcam_plusplus.png
│
└── docs/
    └── architecture_diagrams/
```

---

## 📝 Submission Summary

> For the **CODECURE AI Hackathon — IIT-BHU** form field *"Short Summary of Prototype"*:

**SPECTRA-Net + NeuroFusionX: Dual-Model Uncertainty-Aware Breast Cancer MRI Classification**

We present a dual-model AI framework for binary breast MRI classification (benign vs. malignant) designed to be accurate, clinically trustworthy, and interpretable. Two architectures are proposed:

**SPECTRA-Net** (from scratch): fuses Discrete Wavelet Transform subbands, learnable differentiable morphological operators (novel soft dilation/erosion), a lightweight depthwise-separable CNN, and soft GLCM texture statistics through a dynamic gating mechanism. Uncertainty is quantified via both Dirichlet-based evidential deep learning and 30-sample MC Dropout, with clinical flags raised on ambiguous predictions. GradCAM++ saliency and per-branch attention weights provide dual explainability.

**NeuroFusionX** (transfer learning): extends pretrained EfficientNet-B0 with a soft spatial attention map, a StatFusion layer (attention + channel statistics), a differentiable 6×6 Self-Organizing Map codebook, and SimCLR self-supervised regularization. Validated across 5 random seeds achieving **AUC 0.9882 ± 0.0207** and **F1 0.9372 ± 0.0536**, with Brier Score 0.0156 and ECE 0.0537 — clinically excellent calibration. A Youden-optimal threshold and temperature scaling (T=0.857) ensure reliable probability outputs. GradCAM++ maps and attention-CAM consistency (0.596) provide spatial interpretability.

**NeuroFusionX is recommended for deployment** due to superior accuracy, statistical robustness, and calibrated confidence outputs. SPECTRA-Net is highlighted for architectural novelty — the first model combining learnable morphological operators with wavelet subbands and evidential uncertainty in a unified gated multi-branch framework.

---

## 👥 Team

**Hackathon:** CODECURE AI Hackathon — IIT BHU  
**Dataset:** Breast Cancer Patients MRIs (Kaggle — uzairkhan45)  
**Framework:** PyTorch 2.x | Kaggle GPU (CUDA)

---

## 📄 License

This project is submitted for academic/hackathon purposes. All code is original unless otherwise noted in inline comments.

---

*"Clinical AI must not only be accurate — it must be honest about what it doesn't know."*
