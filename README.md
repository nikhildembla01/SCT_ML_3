# 🐾 Dog vs Cat Image Classification using SVM

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

A **production-ready Machine Learning web application** that classifies images as either a **Dog 🐶 or Cat 🐱** using a Support Vector Machine (SVM). Built with Python, Streamlit, and scikit-learn — ready for your GitHub, LinkedIn, or internship portfolio.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 SVM Classifier | RBF-kernel SVM with probability calibration (Platt scaling) |
| ⚙️ Image Preprocessing | Grayscale conversion, 64×64 resize, StandardScaler normalization |
| 📊 Interactive Dashboard | 8-page Streamlit dashboard with dark theme and modern UI |
| 🔍 Live Prediction | Upload any image → instant label + confidence score |
| 📈 Evaluation Metrics | Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report |
| 📉 Visualizations | Distribution charts, CM heatmap, metrics comparison bar chart |
| ⬇️ Downloads | Export trained model (`.pkl`) and metrics report (`.json`) |
| 💾 Model Persistence | Auto-load saved model on startup with `pickle` |

---

## 🗂️ Project Structure

```
Dog_Cat_Classification/
│
├── app.py              ← Streamlit dashboard (8 pages)
├── train_model.py      ← End-to-end training pipeline
├── model.pkl           ← Saved model bundle (created after training)
├── metrics.json        ← Evaluation metrics (created after training)
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
│
├── dataset/
│   ├── cats/           ← Place cat images here (.jpg / .png)
│   └── dogs/           ← Place dog images here (.jpg / .png)
│
├── images/             ← App screenshots (optional)
└── assets/             ← Additional assets (optional)
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/dog-cat-svm.git
cd dog-cat-svm
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Dataset Images

Download the [Dogs vs. Cats dataset from Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data) and place images in:

```
dataset/cats/   ← cat.0.jpg, cat.1.jpg, …
dataset/dogs/   ← dog.0.jpg, dog.1.jpg, …
```

> **Tip:** Start with 200–500 images per class for a fast demo; use all 12 500 for production accuracy.

### 5. Train the Model

```bash
python train_model.py
```

This outputs `model.pkl` and `metrics.json`.

### 6. Launch the Dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🔬 How It Works

### Image Preprocessing Pipeline

```
Raw Image → Grayscale → Resize 64×64 → Flatten → StandardScaler → SVM Input
```

| Step | Detail |
|------|--------|
| Load | `cv2.imread()` → BGR array |
| Grayscale | `cv2.cvtColor(img, COLOR_BGR2GRAY)` — 3 channels → 1 |
| Resize | `cv2.resize(gray, (64,64))` — fixed 4 096 features |
| Flatten | `array.flatten()` — 2D → 1D vector |
| Normalize | `StandardScaler` — zero mean, unit variance |

### Support Vector Machine (SVM)

SVM finds the **optimal hyperplane** that maximally separates two classes in feature space.

- **Kernel:** RBF (Radial Basis Function) — maps data to higher dimensions to handle non-linear patterns
- **C = 10:** Controls regularization; higher C → tighter fit, lower C → wider margin
- **Gamma = 'scale':** Controls influence radius of each training point
- **Probability = True:** Platt scaling adds calibrated confidence scores

### Why Grayscale + Flatten?

Grayscale reduces dimensionality 3× (RGB → single channel). At 64×64, each image becomes a **4 096-dimensional vector** — compact enough for SVM but rich enough for reasonable accuracy.

---

## 📊 Technologies Used

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Core language |
| Streamlit | ≥1.35 | Interactive web dashboard |
| scikit-learn | ≥1.3 | SVM, metrics, preprocessing |
| OpenCV | ≥4.8 | Image loading and processing |
| NumPy | ≥1.24 | Numerical computations |
| Pandas | ≥2.0 | Data tables |
| Matplotlib | ≥3.7 | Charts and plots |
| Seaborn | ≥0.12 | Confusion matrix heatmap |
| Pillow | ≥10.0 | Image handling in Streamlit |
| pickle | stdlib | Model serialization |

---

## 📈 Expected Performance

| Metric | 200 img/class | 500 img/class | 1000 img/class |
|--------|:---:|:---:|:---:|
| Accuracy | ~68% | ~74% | ~80% |
| Precision | ~67% | ~73% | ~79% |
| Recall | ~68% | ~74% | ~80% |
| F1 Score | ~67% | ~73% | ~79% |

> Performance improves significantly with more training data.

---

## 🚧 Future Improvements

- [ ] **CNN / Transfer Learning** (ResNet, VGG16) for 95%+ accuracy
- [ ] **HOG Features** instead of raw pixels for better texture representation
- [ ] **Grid Search / Cross-Validation** for automatic hyperparameter tuning
- [ ] **Multi-class support** (extend beyond cats and dogs)
- [ ] **Real-time webcam prediction**
- [ ] **Model versioning** with MLflow or DVC
- [ ] **Docker containerisation** for one-command deployment
- [ ] **REST API** with FastAPI for programmatic access

---

## 📜 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.

---

## 🙌 Acknowledgements

- [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) competition
- [scikit-learn](https://scikit-learn.org/) documentation
- [Streamlit](https://streamlit.io/) community

---

*Made with ❤️ for learning ML and building portfolio projects.*
