"""
Train SVM model for Dog vs Cat Classification
==============================================
This script downloads a sample dataset, preprocesses images,
trains an SVM classifier, and saves the model.
"""

import os
import pickle
import numpy as np
import urllib.request
import zipfile
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ─── Config ─────────────────────────────────────────────────────────────────
IMG_SIZE    = (64, 64)          # Resize all images to 64×64
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "svm_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LABELS_PATH = os.path.join(MODEL_DIR, "label_map.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.pkl")
DATASET_DIR  = "dataset"

LABEL_MAP = {0: "Cat", 1: "Dog"}

# ─── Helpers ────────────────────────────────────────────────────────────────

def preprocess_image(img_path: str, size=IMG_SIZE) -> np.ndarray | None:
    """Load an image, convert to RGB, resize, and flatten to 1-D feature vector."""
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(size)
        arr = np.array(img, dtype=np.float32) / 255.0   # normalise to [0, 1]
        return arr.flatten()
    except Exception as e:
        print(f"  ⚠  Skipping {img_path}: {e}")
        return None


def load_dataset(dataset_dir: str):
    """Walk dataset_dir expecting sub-folders 'cats/' and 'dogs/'."""
    X, y = [], []
    class_dirs = {"cats": 0, "dogs": 1}

    for cls_name, label in class_dirs.items():
        folder = os.path.join(dataset_dir, cls_name)
        if not os.path.isdir(folder):
            print(f"  ⚠  Folder not found: {folder}")
            continue
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"  📂  Loading {len(files)} images from '{cls_name}/'")
        for fname in files:
            feat = preprocess_image(os.path.join(folder, fname))
            if feat is not None:
                X.append(feat)
                y.append(label)

    return np.array(X), np.array(y)


def generate_synthetic_dataset(n_per_class: int = 300):
    """
    Generate a tiny synthetic dataset so the app works without Kaggle credentials.
    Dogs  → brighter, warmer hues (high R, medium G/B)
    Cats  → cooler, darker hues  (low R, higher B)
    """
    print("  🔧  Generating synthetic dog/cat dataset …")
    rng = np.random.default_rng(42)
    dim = IMG_SIZE[0] * IMG_SIZE[1] * 3

    # Dogs: high red channel, moderate green, lower blue
    dogs = rng.normal(loc=[0.65, 0.50, 0.35], scale=0.12,
                      size=(n_per_class, 3))
    dogs = np.clip(dogs, 0, 1)
    X_dogs = np.tile(dogs, IMG_SIZE[0] * IMG_SIZE[1]).reshape(n_per_class, dim)
    X_dogs += rng.normal(0, 0.05, X_dogs.shape)

    # Cats: lower red, higher blue
    cats = rng.normal(loc=[0.35, 0.45, 0.60], scale=0.12,
                      size=(n_per_class, 3))
    cats = np.clip(cats, 0, 1)
    X_cats = np.tile(cats, IMG_SIZE[0] * IMG_SIZE[1]).reshape(n_per_class, dim)
    X_cats += rng.normal(0, 0.05, X_cats.shape)

    X = np.vstack([X_dogs, X_cats]).astype(np.float32)
    X = np.clip(X, 0, 1)
    y = np.array([1] * n_per_class + [0] * n_per_class)

    # Persist as images so the rest of the pipeline is identical
    for label_name, arr in [("dogs", X_dogs), ("cats", X_cats)]:
        folder = os.path.join(DATASET_DIR, label_name)
        os.makedirs(folder, exist_ok=True)
        label_int = 1 if label_name == "dogs" else 0
        for i, feat in enumerate(arr):
            px = (np.clip(feat[:dim], 0, 1) * 255).astype(np.uint8)
            img = Image.fromarray(px.reshape(*IMG_SIZE, 3), "RGB")
            img.save(os.path.join(folder, f"{label_name}_{i:04d}.jpg"))

    print(f"  ✅  Synthetic dataset saved to '{DATASET_DIR}/'")
    return X, y


# ─── Main ────────────────────────────────────────────────────────────────────

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load or generate data
    cats_dir = os.path.join(DATASET_DIR, "cats")
    dogs_dir = os.path.join(DATASET_DIR, "dogs")

    if os.path.isdir(cats_dir) and os.path.isdir(dogs_dir):
        print("📂  Loading existing dataset …")
        X, y = load_dataset(DATASET_DIR)
        if len(X) == 0:
            print("  Dataset folders are empty – falling back to synthetic data.")
            X, y = generate_synthetic_dataset()
    else:
        X, y = generate_synthetic_dataset()

    print(f"\n📊  Dataset: {len(X)} samples | Dogs: {(y==1).sum()} | Cats: {(y==0).sum()}")

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"🔀  Train: {len(X_train)}  |  Test: {len(X_test)}")

    # 3. Feature scaling
    print("⚙️   Scaling features …")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 4. Train SVM
    print("🤖  Training SVM (RBF kernel, C=10) …")
    model = SVC(kernel="rbf", C=10, gamma="scale",
                probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    print("  ✅  Training complete.")

    # 5. Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec  = recall_score(y_test, y_pred, average="weighted")
    f1   = f1_score(y_test, y_pred, average="weighted")
    cm   = confusion_matrix(y_test, y_pred)

    print("\n" + "="*45)
    print("  MODEL EVALUATION")
    print("="*45)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

    # 6. Save artefacts
    metrics = {
        "accuracy": acc, "precision": prec,
        "recall": rec, "f1_score": f1,
        "confusion_matrix": cm,
        "y_test": y_test, "y_pred": y_pred,
        "train_size": len(X_train), "test_size": len(X_test),
        "total_samples": len(X),
        "dogs_count": int((y == 1).sum()),
        "cats_count": int((y == 0).sum()),
    }

    with open(MODEL_PATH,  "wb") as f: pickle.dump(model,     f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler,    f)
    with open(LABELS_PATH, "wb") as f: pickle.dump(LABEL_MAP, f)
    with open(METRICS_PATH,"wb") as f: pickle.dump(metrics,   f)

    print(f"\n💾  Model  saved → {MODEL_PATH}")
    print(f"💾  Scaler saved → {SCALER_PATH}")
    print(f"💾  Metrics saved → {METRICS_PATH}")
    print("\n🎉  Training pipeline complete!")


if __name__ == "__main__":
    train()