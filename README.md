# 🕵️‍♂️ Deepfake Detection App

A Streamlit web app that detects deepfake videos using an **ensemble of CLIP ViT-L/14 and EfficientNet-B4** models — trained on FaceForensics++ (c23) and Celeb-DF v2.

---

## ✨ What's New (v2)

| | Old (GRU) | New (Ensemble) |
|---|---|---|
| **Architecture** | Xception features → GRU | CLIP ViT-L/14 (LayerNorm + SVD adapters) + EfficientNet-B4 |
| **Ensemble AUC** | — | **1.000** (CelebDF), **0.989** (FF++) |
| **Zero-shot AUC** | — | 0.970 (CLIP-SVD on unseen CelebDF) |
| **Model file** | `gru_model.h5` + `evaluation_metrics.pkl` | `deepfake_detector.pkl` (all weights bundled, ~716 MB) |
| **Face detection** | None (raw frames) | MTCNN face cropping per frame |
| **Input types** | Video only | Video + face images (JPG/PNG) |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install streamlit torch torchvision transformers timm facenet-pytorch \
            opencv-python scikit-learn pandas numpy pillow scipy
```

> **GPU recommended** (CUDA). CPU works but is slower (~10–30 s per clip at 16 frames).

### 2. Place the model bundle

Put `deepfake_detector.pkl` in the same directory as `Streamlit.py`.  
*(~716 MB — download from the Kaggle notebook output or the project release.)*

### 3. Run

```bash
streamlit run Streamlit.py
```

Then open <http://localhost:8501> in your browser.

---

## 🧠 Model Details

### Architecture

The detector is a **weighted ensemble** of four models trained on MTCNN-cropped face images:

| Model | Backbone | Training data | Notes |
|-------|----------|---------------|-------|
| `F_clip` | CLIP ViT-L/14 (LayerNorm-only tuning) | FF++ + CelebDF | Main CLIP variant |
| `F_svd` | CLIP ViT-L/14 (SVD/Effort-style adapters) | FF++ + CelebDF | Rank-1 residual on every attention projection |
| `F_eff` | EfficientNet-B4 | FF++ + CelebDF | CNN complement |
| `F_eff_s1` | EfficientNet-B4 (seed 1) | FF++ + CelebDF | CNN ensemble diversity |

Ensemble weights are **fitted on the validation set** to maximise balanced accuracy, with EfficientNet share capped at 30 %.

### Key Metrics (Test Set)

| Split | Videos | AUC | Accuracy | F1 | MCC |
|-------|--------|-----|----------|----|-----|
| ENSEMBLE \| CelebDF | 508 | **1.000** | 99.6 % | 0.998 | 0.986 |
| ENSEMBLE \| FF++ | 648 | **0.989** | 94.9 % | 0.969 | 0.837 |
| CLIP-SVD (zero-shot → CelebDF) | 3 889 | 0.970 | 92.9 % | 0.953 | 0.801 |

### Pipeline

```
Video / Image
    └─► MTCNN face detection (per frame)
         └─► Face crop (256 × 256, margin 1.3×)
              └─► Resize per backbone (224 for CLIP, 256 for EfficientNet)
                   └─► Model forward pass (optional TTA: H-flip average)
                        └─► Weighted ensemble → calibrated prob_fake
```

---

## 📁 Project Structure

```
DeepFake/
├── Streamlit.py              # Streamlit web app
├── dfd_lib.py                # Pipeline library (Detector class, training, evaluation)
├── deepfake_detector.pkl     # Bundled model weights + ensemble config (~716 MB)
├── final_metrics.csv         # Per-set evaluation metrics
├── requirements.txt          # Python dependencies
├── LICENSE
└── README.md
```

---

## 📦 Requirements

```
streamlit>=1.28
torch>=2.0
torchvision
transformers>=4.37
timm>=0.9
facenet-pytorch
opencv-python
scikit-learn
pandas
numpy
pillow
scipy
```

---

## 🗒️ Training (Reproducibility)

The full training pipeline is documented in `New/deepfake (2).ipynb` (Kaggle 2× T4, ~3.5 h).  
Key settings:

- **Face extraction:** MTCNN, 12–16 frames per video
- **Splits:** identity-group based (no person shared between train / val / test)
- **Data:** FF++ c23 (1 000 real, 5 methods × 1 000 fake) + Celeb-DF v2 (890 real, 5 639 fake)
- **CLIP tuning:** LayerNorm-only (ClipLN) and rank-1 SVD adapter (ClipSVD / Effort-style)
- **Loss:** BCE with label smoothing (0.025) + optional uniformity/alignment regularisation

---

## 📄 License

MIT — see [LICENSE](LICENSE).
