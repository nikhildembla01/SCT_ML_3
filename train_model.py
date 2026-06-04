# -*- coding: utf-8 -*-
"""
train_model.py
==============
Dog vs Cat Image Classification using SVM
Trains the SVM model on the dataset and saves it as model.pkl
"""

import os
import cv2
import numpy as np
import pickle
import json
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────
IMAGE_SIZE   = (64, 64)
CATS_DIR     = "dataset/cats"
DOGS_DIR     = "dataset/dogs"
MODEL_PATH   = "model.pkl"
METRICS_PATH = "metrics.json"
TEST_SIZE    = 0.2
RANDOM_STATE = 42
MAX_IMAGES   = 500
VALID_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_and_preprocess(image_path, size=IMAGE_SIZE):
    """Load image, convert to grayscale, resize, flatten."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size)
    return resized.flatten().astype(np.float32)


def load_dataset(cats_dir=CATS_DIR, dogs_dir=DOGS_DIR, max_per_class=MAX_IMAGES):
    """Read images from cats/ and dogs/ folders. Labels: 0=Cat, 1=Dog"""
    X, y = [], []

    def read_class(folder, label, name):
        if not os.path.exists(folder):
            print(f"  [WARN] Folder not found: {folder}")
            return
        files = [
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in VALID_EXTS
        ][:max_per_class]
        for fname in files:
            features = load_and_preprocess(os.path.join(folder, fname))
            if features is not None:
                X.append(features)
                y.append(label)
        print(f"  Loaded {len([f for f in files])} {name} images")

    print("\n[1/5] Loading dataset...")
    read_class(cats_dir, 0, "cat")
    read_class(dogs_dir, 1, "dog")

    if not X:
        raise ValueError("No images found! Add images to dataset/cats/ and dataset/dogs/")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def normalize(X_train, X_test):
    """StandardScaler normalization - fit only on training data."""
    print("\n[2/5] Normalising pixel values...")
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler


def train_svm(X_train, y_train):
    """Train RBF-kernel SVM with probability estimates."""
    print("\n[3/5] Training SVM classifier...")
    clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    print("  SVM training complete!")
    return clf


def evaluate(clf, X_test, y_test):
    """Compute evaluation metrics."""
    print("\n[4/5] Evaluating model...")
    y_pred = clf.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy":         round(acc  * 100, 2),
        "precision":        round(prec * 100, 2),
        "recall":           round(rec  * 100, 2),
        "f1_score":         round(f1   * 100, 2),
        "confusion_matrix": cm,
        "report": classification_report(y_test, y_pred, target_names=["Cat", "Dog"]),
        "test_size":  len(y_test),
        "image_size": list(IMAGE_SIZE),
    }
    print(f"  Accuracy: {acc*100:.2f}%  Precision: {prec*100:.2f}%  Recall: {rec*100:.2f}%  F1: {f1*100:.2f}%")
    return metrics


def save_artifacts(clf, scaler, metrics):
    """Save model bundle and metrics."""
    print("\n[5/5] Saving model & metrics...")
    bundle = {"model": clf, "scaler": scaler, "image_size": IMAGE_SIZE}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {MODEL_PATH}  &  {METRICS_PATH}")


def main():
    print("=" * 55)
    print("  Dog vs Cat Classification — SVM Training Pipeline")
    print("=" * 55)

    X, y = load_dataset(CATS_DIR, DOGS_DIR)
    print(f"\n  Total: {len(X)} | Cats: {np.sum(y==0)} | Dogs: {np.sum(y==1)} | Features: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_tr_s, X_te_s, scaler = normalize(X_train, X_test)
    clf     = train_svm(X_tr_s, y_train)
    metrics = evaluate(clf, X_te_s, y_test)
    save_artifacts(clf, scaler, metrics)

    print("\nDone! Run: streamlit run app.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
