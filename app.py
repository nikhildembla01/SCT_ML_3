# -*- coding: utf-8 -*-
"""
app.py  —  Dog vs Cat Image Classifier  |  SVM + Streamlit Dashboard
=====================================================================
Run: streamlit run app.py
"""

# ─── Standard library ────────────────────────────────────────────────────────
import os, io, json, pickle, base64, datetime, warnings
warnings.filterwarnings("ignore")

# ─── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
IMAGE_SIZE   = (64, 64)
CATS_DIR     = "dataset/cats"
DOGS_DIR     = "dataset/dogs"
MODEL_PATH   = "model.pkl"
METRICS_PATH = "metrics.json"
TEST_SIZE    = 0.2
RANDOM_STATE = 42
MAX_IMAGES   = 500
VALID_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

LABEL_MAP = {0: "🐱 Cat", 1: "🐶 Dog"}
LABEL_RAW = {0: "Cat", 1: "Dog"}

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dog vs Cat Classifier · SVM",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- Google fonts ---------- */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ---------- Root tokens ---------- */
:root {
  --bg:       #0d0f14;
  --surface:  #161922;
  --surface2: #1e2330;
  --border:   #2a3045;
  --accent1:  #6c63ff;
  --accent2:  #ff6584;
  --accent3:  #43e97b;
  --accent4:  #f7971e;
  --text:     #e8ecf4;
  --muted:    #8892a4;
  --cat:      #ff6584;
  --dog:      #6c63ff;
  --radius:   14px;
  --shadow:   0 4px 24px rgba(0,0,0,.45);
}

/* ---------- Base ---------- */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}

/* Hide default streamlit header decorations */
#MainMenu, footer, header { visibility: hidden; }

/* ---------- Typography ---------- */
h1,h2,h3,h4,h5,h6 {
  font-family: 'Syne', sans-serif !important;
  color: var(--text) !important;
}

p, li, label, div { color: var(--text); }

/* ---------- Sidebar headings ---------- */
.sidebar-logo {
  font-family: 'Syne', sans-serif;
  font-size: 1.6rem;
  font-weight: 800;
  background: linear-gradient(135deg, var(--accent1), var(--accent2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 4px;
}
.sidebar-sub {
  font-size: .78rem;
  color: var(--muted);
  letter-spacing: .06em;
  text-transform: uppercase;
  margin-bottom: 24px;
}

/* ---------- Nav items ---------- */
.nav-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  border-radius: 10px;
  cursor: pointer;
  font-size: .9rem;
  font-weight: 500;
  color: var(--muted);
  transition: all .2s;
  margin-bottom: 4px;
  border: 1px solid transparent;
}
.nav-item:hover { background: var(--surface2); color: var(--text); }
.nav-item.active {
  background: linear-gradient(135deg,rgba(108,99,255,.18),rgba(255,101,132,.10));
  color: var(--text);
  border-color: rgba(108,99,255,.3);
}

/* ---------- Hero banner ---------- */
.hero {
  background: linear-gradient(135deg, #1a1c2e 0%, #1a1228 50%, #0d1820 100%);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 40px 48px;
  margin-bottom: 28px;
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: "";
  position: absolute; inset: 0;
  background:
    radial-gradient(ellipse 60% 50% at 80% 50%, rgba(108,99,255,.12), transparent),
    radial-gradient(ellipse 40% 60% at 20% 60%, rgba(255,101,132,.08), transparent);
  pointer-events: none;
}
.hero-badge {
  display: inline-block;
  font-size: .72rem;
  font-weight: 600;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--accent1);
  background: rgba(108,99,255,.15);
  border: 1px solid rgba(108,99,255,.3);
  padding: 4px 12px;
  border-radius: 100px;
  margin-bottom: 18px;
}
.hero-title {
  font-family: 'Syne', sans-serif;
  font-size: 2.6rem;
  font-weight: 800;
  line-height: 1.15;
  color: var(--text) !important;
  margin-bottom: 14px;
}
.hero-title span { color: var(--accent1); }
.hero-desc {
  font-size: .98rem;
  color: var(--muted);
  max-width: 620px;
  line-height: 1.7;
}
.hero-pills {
  display: flex; flex-wrap: wrap; gap: 8px;
  margin-top: 22px;
}
.pill {
  font-size: .75rem;
  font-weight: 600;
  letter-spacing: .05em;
  padding: 5px 14px;
  border-radius: 100px;
  border: 1px solid var(--border);
  color: var(--muted);
  background: var(--surface2);
}

/* ---------- Cards ---------- */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 22px 24px;
  margin-bottom: 16px;
  position: relative;
  overflow: hidden;
  transition: border-color .2s, box-shadow .2s;
}
.card:hover {
  border-color: rgba(108,99,255,.35);
  box-shadow: 0 0 0 1px rgba(108,99,255,.1);
}
.card-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 6px;
}
.card-body { font-size: .88rem; color: var(--muted); line-height: 1.6; }

/* ---------- Metric tiles ---------- */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 14px;
  margin-bottom: 24px;
}
.metric-tile {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 20px;
  transition: transform .2s, border-color .2s;
}
.metric-tile:hover { transform: translateY(-2px); border-color: rgba(108,99,255,.4); }
.metric-label {
  font-size: .72rem;
  text-transform: uppercase;
  letter-spacing: .1em;
  color: var(--muted);
  margin-bottom: 8px;
}
.metric-value {
  font-family: 'Syne', sans-serif;
  font-size: 2rem;
  font-weight: 800;
  line-height: 1;
}
.metric-unit {
  font-size: .8rem;
  color: var(--muted);
  margin-top: 4px;
}

/* ---------- Section headings ---------- */
.section-head {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 32px 0 20px;
}
.section-icon {
  width: 36px; height: 36px;
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.1rem;
  flex-shrink: 0;
}
.section-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.3rem;
  font-weight: 700;
  color: var(--text);
}
.section-desc { font-size: .82rem; color: var(--muted); margin-top: 2px; }

/* ---------- Prediction result box ---------- */
.pred-box {
  border-radius: var(--radius);
  padding: 28px 32px;
  text-align: center;
  margin-top: 20px;
  position: relative;
  overflow: hidden;
}
.pred-box.cat {
  background: linear-gradient(135deg, rgba(255,101,132,.12), rgba(255,101,132,.04));
  border: 1px solid rgba(255,101,132,.35);
}
.pred-box.dog {
  background: linear-gradient(135deg, rgba(108,99,255,.12), rgba(108,99,255,.04));
  border: 1px solid rgba(108,99,255,.35);
}
.pred-animal { font-size: 4.5rem; margin-bottom: 10px; }
.pred-label {
  font-family: 'Syne', sans-serif;
  font-size: 2rem;
  font-weight: 800;
  color: var(--text);
}
.pred-conf {
  font-size: .9rem;
  color: var(--muted);
  margin-top: 6px;
}
.conf-bar-wrap {
  background: var(--surface2);
  border-radius: 100px;
  height: 8px;
  margin: 14px auto;
  max-width: 280px;
  overflow: hidden;
}
.conf-bar {
  height: 100%;
  border-radius: 100px;
  transition: width .6s ease;
}

/* ---------- Upload zone ---------- */
[data-testid="stFileUploader"] {
  background: var(--surface2) !important;
  border: 2px dashed var(--border) !important;
  border-radius: var(--radius) !important;
  transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--accent1) !important;
}

/* ---------- Buttons ---------- */
[data-testid="stButton"] button {
  background: linear-gradient(135deg, var(--accent1), #8b5cf6) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  padding: 10px 24px !important;
  transition: opacity .2s, transform .2s !important;
}
[data-testid="stButton"] button:hover {
  opacity: .88 !important;
  transform: translateY(-1px) !important;
}

/* ---------- Selectbox / radio ---------- */
[data-testid="stSelectbox"] label,
[data-testid="stRadio"] label { color: var(--text) !important; }

/* ---------- Tabs ---------- */
[data-testid="stTabs"] [role="tab"] {
  color: var(--muted) !important;
  font-family: 'DM Sans', sans-serif;
  font-weight: 500;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color: var(--text) !important;
  border-bottom: 2px solid var(--accent1) !important;
}

/* ---------- Info / success / warning boxes ---------- */
[data-testid="stAlert"] {
  background: var(--surface2) !important;
  border-left-color: var(--accent1) !important;
  color: var(--text) !important;
}

/* ---------- DataFrame ---------- */
[data-testid="stDataFrame"] { border-radius: var(--radius); overflow: hidden; }

/* ---------- Progress bar ---------- */
[data-testid="stProgress"] > div > div {
  background: linear-gradient(90deg, var(--accent1), var(--accent2)) !important;
}

/* ---------- Divider ---------- */
.divider {
  border: none;
  border-top: 1px solid var(--border);
  margin: 28px 0;
}

/* ---------- Code blocks ---------- */
code { color: var(--accent3) !important; background: var(--surface2) !important; }
pre  { background: var(--surface2) !important; border: 1px solid var(--border) !important; }

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════

def preprocess_image(img_array, size=IMAGE_SIZE):
    """Convert image array → grayscale → resize → flatten → float32."""
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    resized = cv2.resize(gray, size)
    return resized.flatten().astype(np.float32)


@st.cache_data
def load_sample_images(folder, n=6):
    """Return up to n PIL images from a folder (cached)."""
    imgs = []
    if not os.path.exists(folder):
        return imgs
    files = [f for f in os.listdir(folder)
             if os.path.splitext(f)[1].lower() in VALID_EXTS][:n]
    for f in files:
        try:
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            img.thumbnail((200, 200))
            imgs.append(img)
        except Exception:
            pass
    return imgs


@st.cache_data
def load_dataset_stats():
    """Count images per class, return dict."""
    cats = dogs = 0
    if os.path.exists(CATS_DIR):
        cats = sum(1 for f in os.listdir(CATS_DIR)
                   if os.path.splitext(f)[1].lower() in VALID_EXTS)
    if os.path.exists(DOGS_DIR):
        dogs = sum(1 for f in os.listdir(DOGS_DIR)
                   if os.path.splitext(f)[1].lower() in VALID_EXTS)
    return {"cats": cats, "dogs": dogs, "total": cats + dogs}


def load_model():
    """Load pickled model bundle; return (clf, scaler, img_size) or None."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        return bundle.get("model"), bundle.get("scaler"), bundle.get("image_size", IMAGE_SIZE)
    return None, None, IMAGE_SIZE


def load_metrics():
    """Load saved metrics JSON or return None."""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


def fig_to_img(fig):
    """Convert matplotlib figure to bytes buffer."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=130)
    buf.seek(0)
    return buf


def accent(color):
    """Return hex for named accent."""
    MAP = {"purple": "#6c63ff", "pink": "#ff6584",
           "green": "#43e97b", "orange": "#f7971e"}
    return MAP.get(color, color)


# ════════════════════════════════════════════════════════
# CHART HELPERS  (all use dark theme)
# ════════════════════════════════════════════════════════
BG  = "#0d0f14"
S1  = "#161922"
S2  = "#1e2330"
BDR = "#2a3045"
TXT = "#e8ecf4"
MUT = "#8892a4"

def _dark_fig(w=8, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(S1)
    for sp in ax.spines.values():
        sp.set_color(BDR)
    ax.tick_params(colors=MUT, labelsize=9)
    ax.xaxis.label.set_color(MUT)
    ax.yaxis.label.set_color(MUT)
    ax.title.set_color(TXT)
    return fig, ax


def chart_class_distribution(stats):
    fig, ax = _dark_fig(6, 3.8)
    labels  = ["Cats", "Dogs"]
    values  = [stats["cats"], stats["dogs"]]
    colors  = [accent("pink"), accent("purple")]
    bars    = ax.bar(labels, values, color=colors, width=.5,
                     edgecolor=BDR, linewidth=.8)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2,
                str(int(b.get_height())), ha="center", va="bottom",
                color=TXT, fontsize=10, fontweight="bold")
    ax.set_title("Class Distribution", fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("Image Count")
    ax.grid(axis="y", color=BDR, linewidth=.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def chart_confusion_matrix(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
    ax.set_facecolor(BG)
    cmap = sns.diverging_palette(260, 345, s=80, l=40, as_cmap=True)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"],
        ax=ax, linewidths=1.5, linecolor=BG,
        annot_kws={"size": 14, "weight": "bold", "color": TXT},
        cbar_kws={"shrink": .7}
    )
    ax.tick_params(colors=TXT, labelsize=10)
    ax.set_xlabel("Predicted", color=MUT, fontsize=10)
    ax.set_ylabel("Actual",    color=MUT, fontsize=10)
    ax.set_title(title, color=TXT, fontsize=12, fontweight="bold", pad=14)
    ax.yaxis.set_tick_params(rotation=0)
    plt.tight_layout()
    return fig


def chart_metrics_radar(metrics):
    """Bar chart comparing all four metrics."""
    fig, ax = _dark_fig(7, 3.8)
    keys   = ["Accuracy", "Precision", "Recall", "F1 Score"]
    vals   = [metrics["accuracy"], metrics["precision"],
               metrics["recall"],   metrics["f1_score"]]
    colors = [accent("purple"), accent("pink"), accent("green"), accent("orange")]
    bars   = ax.bar(keys, vals, color=colors, width=.5, edgecolor=BDR, linewidth=.8)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + .4,
                f"{b.get_height():.1f}%", ha="center", va="bottom",
                color=TXT, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Evaluation Metrics", fontsize=13, fontweight="bold", pad=14)
    ax.grid(axis="y", color=BDR, linewidth=.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def chart_preprocessing_pipeline():
    """Diagram showing preprocessing steps."""
    fig = plt.figure(figsize=(10, 2.2), facecolor=BG)
    steps = ["Original\nImage", "Grayscale\nConversion", "Resize\n64×64",
             "Flatten\n→ 4096-d", "Normalize\n(StandardScaler)", "SVM\nInput"]
    colors = ["#6c63ff","#8b5cf6","#ff6584","#f7971e","#43e97b","#38bdf8"]
    n = len(steps)
    for i, (s, c) in enumerate(zip(steps, colors)):
        x = i / (n - 1)
        fig.text(x, .5, s, ha="center", va="center", fontsize=9.5,
                 fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=.55", facecolor=c,
                           edgecolor="none", alpha=.85))
        if i < n - 1:
            fig.text((x + (i+1)/(n-1))/2, .5, "→", ha="center",
                     va="center", fontsize=14, color=MUT)
    fig.text(.5, .08, "Image Preprocessing Pipeline", ha="center",
             fontsize=9, color=MUT)
    return fig


# ════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════
PAGES = [
    ("🏠", "Home",           "Overview & project info"),
    ("📊", "Dataset",        "Browse & explore images"),
    ("⚙️", "Preprocessing", "Feature engineering steps"),
    ("🧠", "Model Training", "Train / retrain SVM"),
    ("📈", "Evaluation",     "Metrics & confusion matrix"),
    ("🔍", "Prediction",     "Classify your own image"),
    ("📉", "Visualizations", "Charts & analysis"),
    ("⬇️", "Downloads",     "Export model & results"),
]

with st.sidebar:
    st.markdown('<div class="sidebar-logo">🐾 PawClassify</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">SVM · Dog vs Cat</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [p[1] for p in PAGES],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#2a3045;margin:20px 0'>", unsafe_allow_html=True)

    # Model status badge
    clf, scaler, img_size = load_model()
    if clf:
        st.markdown("""
        <div style='background:rgba(67,233,123,.1);border:1px solid rgba(67,233,123,.3);
             border-radius:10px;padding:12px 14px;'>
          <div style='font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;
               color:#43e97b;font-weight:600;margin-bottom:4px;'>Model Status</div>
          <div style='color:#e8ecf4;font-size:.88rem;font-weight:500;'>✅ Loaded & Ready</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:rgba(247,151,30,.08);border:1px solid rgba(247,151,30,.3);
             border-radius:10px;padding:12px 14px;'>
          <div style='font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;
               color:#f7971e;font-weight:600;margin-bottom:4px;'>Model Status</div>
          <div style='color:#e8ecf4;font-size:.88rem;font-weight:500;'>⚠️ Not trained yet</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2a3045;margin:20px 0'>", unsafe_allow_html=True)
    stats = load_dataset_stats()
    st.markdown(f"""
    <div style='font-size:.75rem;color:#8892a4;'>
      <div style='margin-bottom:6px;'>📁 <b style='color:#e8ecf4'>{stats['total']}</b> images loaded</div>
      <div style='margin-bottom:6px;'>🐱 Cats: <b style='color:#ff6584'>{stats['cats']}</b></div>
      <div>🐶 Dogs: <b style='color:#6c63ff'>{stats['dogs']}</b></div>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("""
    <div class="hero">
      <div class="hero-badge">Machine Learning Portfolio Project</div>
      <div class="hero-title">Dog vs Cat<br><span>Image Classification</span></div>
      <div class="hero-desc">
        A production-ready ML pipeline using <strong style='color:#e8ecf4'>Support Vector Machines (SVM)</strong>
        to classify pet images with high accuracy. Built with Python, Streamlit, and scikit-learn.
      </div>
      <div class="hero-pills">
        <span class="pill">Python 3.10+</span>
        <span class="pill">Streamlit</span>
        <span class="pill">scikit-learn SVM</span>
        <span class="pill">OpenCV</span>
        <span class="pill">Pillow</span>
        <span class="pill">NumPy</span>
        <span class="pill">Matplotlib</span>
        <span class="pill">Seaborn</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card">
          <div style='font-size:2rem;margin-bottom:10px;'>🎯</div>
          <div class="card-title">Objective</div>
          <div class="card-body">Binary image classification of dogs and cats using
          SVM with hand-crafted pixel features extracted via OpenCV.</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
          <div style='font-size:2rem;margin-bottom:10px;'>🔬</div>
          <div class="card-title">Methodology</div>
          <div class="card-body">Grayscale conversion → resize to 64×64 →
          flatten → StandardScaler → RBF-kernel SVM with probability calibration.</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card">
          <div style='font-size:2rem;margin-bottom:10px;'>🚀</div>
          <div class="card-title">Deployment</div>
          <div class="card-body">Interactive Streamlit dashboard with live
          prediction, confidence scores, visualisations, and model download.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown("### 📖 How SVM Image Classification Works")
    steps_html = ""
    step_info = [
        ("#6c63ff", "1", "Image Ingestion",
         "Load dog/cat images from disk using OpenCV. Supports JPEG, PNG, BMP, TIFF."),
        ("#ff6584", "2", "Preprocessing",
         "Convert BGR→Grayscale, resize to 64×64 px, flatten to a 4 096-d vector."),
        ("#43e97b", "3", "Normalization",
         "Apply StandardScaler to zero-center and unit-variance each feature."),
        ("#f7971e", "4", "SVM Training",
         "Fit an RBF-kernel SVC on 80% of data. C=10, gamma='scale', probability=True."),
        ("#38bdf8", "5", "Evaluation",
         "Measure accuracy, precision, recall, F1, and plot the confusion matrix."),
        ("#a78bfa", "6", "Prediction",
         "Upload any image → preprocess → predict label + confidence score."),
    ]
    for c, n, title, desc in step_info:
        steps_html += f"""
        <div class="card" style="border-left:3px solid {c};margin-bottom:10px;">
          <div style="display:flex;align-items:flex-start;gap:14px;">
            <div style="background:{c}22;color:{c};border-radius:8px;width:32px;height:32px;
                 display:flex;align-items:center;justify-content:center;
                 font-family:'Syne',sans-serif;font-weight:800;font-size:1rem;flex-shrink:0;">{n}</div>
            <div>
              <div class="card-title" style="margin-bottom:4px;">{title}</div>
              <div class="card-body">{desc}</div>
            </div>
          </div>
        </div>"""
    st.markdown(steps_html, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 🗂️ Quick Start")
    st.code("""# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your images
#    dataset/cats/  ← cat images (jpg/png)
#    dataset/dogs/  ← dog images (jpg/png)

# 3. Train the model
python train_model.py

# 4. Launch dashboard
streamlit run app.py""", language="bash")


# ════════════════════════════════════════════════════════
# PAGE: DATASET
# ════════════════════════════════════════════════════════
elif page == "Dataset":
    st.markdown("""
    <div class="section-head">
      <div class="section-icon" style="background:rgba(108,99,255,.15);">📊</div>
      <div>
        <div class="section-title">Dataset Explorer</div>
        <div class="section-desc">Browse images, class balance, and dataset statistics</div>
      </div>
    </div>""", unsafe_allow_html=True)

    stats = load_dataset_stats()
    balance = (min(stats["cats"], stats["dogs"]) / max(stats["cats"], stats["dogs"]) * 100
               if max(stats["cats"], stats["dogs"]) > 0 else 0)

    c1, c2, c3, c4 = st.columns(4)
    tiles = [
        ("Total Images", str(stats["total"]), "images", "#6c63ff"),
        ("Cat Images",   str(stats["cats"]),  "samples", "#ff6584"),
        ("Dog Images",   str(stats["dogs"]),  "samples", "#6c63ff"),
        ("Balance",      f"{balance:.0f}%",   "class balance", "#43e97b"),
    ]
    for col, (lbl, val, unit, color) in zip([c1,c2,c3,c4], tiles):
        with col:
            st.markdown(f"""
            <div class="metric-tile">
              <div class="metric-label">{lbl}</div>
              <div class="metric-value" style="color:{color};">{val}</div>
              <div class="metric-unit">{unit}</div>
            </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Distribution", "🖼️ Sample Images", "📋 Info"])

    with tab1:
        if stats["total"] > 0:
            fig = chart_class_distribution(stats)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No images found. Add images to dataset/cats/ and dataset/dogs/ folders.")

    with tab2:
        if stats["cats"] > 0 or stats["dogs"] > 0:
            st.markdown("#### 🐱 Cat Samples")
            cat_imgs = load_sample_images(CATS_DIR)
            if cat_imgs:
                cols = st.columns(min(len(cat_imgs), 6))
                for col, img in zip(cols, cat_imgs):
                    with col:
                        st.image(img, use_column_width=True)
            else:
                st.info("No cat images found.")

            st.markdown("#### 🐶 Dog Samples")
            dog_imgs = load_sample_images(DOGS_DIR)
            if dog_imgs:
                cols = st.columns(min(len(dog_imgs), 6))
                for col, img in zip(cols, dog_imgs):
                    with col:
                        st.image(img, use_column_width=True)
            else:
                st.info("No dog images found.")
        else:
            st.info("Add images to dataset/cats/ and dataset/dogs/ to see samples here.")

    with tab3:
        st.markdown("""
        <div class="card">
          <div class="card-title">📦 Dataset Information</div>
          <div class="card-body">
            <p><b style='color:#e8ecf4;'>Source:</b> Dogs vs. Cats — Kaggle Competition (2013)<br>
            <b style='color:#e8ecf4;'>Original Size:</b> 25,000 images (12,500 per class)<br>
            <b style='color:#e8ecf4;'>Image Types:</b> JPEG, PNG, BMP, TIFF, WebP<br>
            <b style='color:#e8ecf4;'>Classes:</b> Binary — Cat (0) vs Dog (1)<br>
            <b style='color:#e8ecf4;'>License:</b> Kaggle competition rules apply</p>
            <p>Place images in <code>dataset/cats/</code> and <code>dataset/dogs/</code>.
            The model uses up to 500 images per class for training speed. You can increase
            <code>MAX_IMAGES</code> in <code>train_model.py</code> for better accuracy.</p>
          </div>
        </div>""", unsafe_allow_html=True)

        if stats["total"] > 0:
            df = pd.DataFrame({
                "Class":  ["Cat", "Dog", "Total"],
                "Count":  [stats["cats"], stats["dogs"], stats["total"]],
                "Share":  [
                    f"{stats['cats']/stats['total']*100:.1f}%" if stats["total"] else "0%",
                    f"{stats['dogs']/stats['total']*100:.1f}%" if stats["total"] else "0%",
                    "100%",
                ],
            })
            st.dataframe(df, use_container_width=True)


# ════════════════════════════════════════════════════════
# PAGE: PREPROCESSING
# ════════════════════════════════════════════════════════
elif page == "Preprocessing":
    st.markdown("""
    <div class="section-head">
      <div class="section-icon" style="background:rgba(247,151,30,.15);">⚙️</div>
      <div>
        <div class="section-title">Data Preprocessing</div>
        <div class="section-desc">How raw images become ML-ready feature vectors</div>
      </div>
    </div>""", unsafe_allow_html=True)

    fig = chart_preprocessing_pipeline()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("<br>", unsafe_allow_html=True)

    steps = [
        ("#6c63ff", "1. Image Loading",
         "Images are read from disk with <code>cv2.imread()</code> in BGR format. "
         "Supported formats: JPG, PNG, BMP, TIFF, WebP.",
         """img = cv2.imread("dog.jpg")  # Loads as BGR uint8 array"""),
        ("#ff6584", "2. Grayscale Conversion",
         "Convert BGR → Grayscale to reduce from 3 channels → 1 channel, "
         "cutting feature size by 3× while preserving structure.",
         """gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # shape: (H, W)"""),
        ("#43e97b", "3. Resizing",
         "All images are resized to 64×64 px so every feature vector has "
         "the same length (4 096). Larger sizes increase accuracy but slow training.",
         """resized = cv2.resize(gray, (64, 64))  # shape: (64, 64)"""),
        ("#f7971e", "4. Flattening",
         "The 2-D pixel array is unrolled into a 1-D vector. "
         "64 × 64 = <b>4 096 features</b> per image.",
         """flat = resized.flatten()  # shape: (4096,)"""),
        ("#38bdf8", "5. StandardScaler Normalization",
         "Zero-mean, unit-variance scaling prevents features with large pixel "
         "values from dominating the SVM kernel. Scaler is fit only on training data.",
         """scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)"""),
    ]

    for color, title, desc, code in steps:
        st.markdown(f"""
        <div class="card" style="border-left:3px solid {color};">
          <div class="card-title" style="color:{color};">{title}</div>
          <div class="card-body">{desc}</div>
        </div>""", unsafe_allow_html=True)
        st.code(code, language="python")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 🔬 Interactive Preprocessing Demo")
    uploaded = st.file_uploader("Upload any image to see preprocessing steps",
                                 type=["jpg","jpeg","png","bmp","webp"])
    if uploaded:
        pil_img  = Image.open(uploaded).convert("RGB")
        img_arr  = np.array(pil_img)
        gray_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        res_arr  = cv2.resize(gray_arr, (64, 64))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Original**")
            st.image(pil_img, use_column_width=True)
            st.caption(f"Shape: {img_arr.shape}")
        with c2:
            st.markdown("**Grayscale**")
            st.image(gray_arr, use_column_width=True, clamp=True)
            st.caption(f"Shape: {gray_arr.shape}")
        with c3:
            st.markdown("**Resized 64×64**")
            st.image(res_arr, use_column_width=True, clamp=True)
            st.caption(f"Shape: {res_arr.shape} → 4096 features")

        flat = res_arr.flatten().astype(np.float32)
        st.markdown(f"""
        <div class="card" style="margin-top:16px;">
          <div class="card-title">Feature Vector Preview</div>
          <div class="card-body">
            Flattened length: <b style='color:#e8ecf4'>{len(flat)}</b> &nbsp;|&nbsp;
            Min pixel: <b style='color:#ff6584'>{flat.min():.0f}</b> &nbsp;|&nbsp;
            Max pixel: <b style='color:#6c63ff'>{flat.max():.0f}</b> &nbsp;|&nbsp;
            Mean: <b style='color:#43e97b'>{flat.mean():.1f}</b>
          </div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# PAGE: MODEL TRAINING
# ════════════════════════════════════════════════════════
elif page == "Model Training":
    st.markdown("""
    <div class="section-head">
      <div class="section-icon" style="background:rgba(67,233,123,.15);">🧠</div>
      <div>
        <div class="section-title">Model Training</div>
        <div class="section-desc">Train or retrain the SVM classifier from the dashboard</div>
      </div>
    </div>""", unsafe_allow_html=True)

    stats = load_dataset_stats()

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown("""
        <div class="card">
          <div class="card-title">🔬 SVM Configuration</div>
          <div class="card-body">
            <table style="width:100%;border-collapse:collapse;">
              <tr><td style="padding:6px 0;color:#8892a4;">Kernel</td>
                  <td style="color:#e8ecf4;font-weight:600;">RBF (Radial Basis Function)</td></tr>
              <tr><td style="padding:6px 0;color:#8892a4;">C (Regularization)</td>
                  <td style="color:#e8ecf4;font-weight:600;">10</td></tr>
              <tr><td style="padding:6px 0;color:#8892a4;">Gamma</td>
                  <td style="color:#e8ecf4;font-weight:600;">scale (1/(n_features × X.var()))</td></tr>
              <tr><td style="padding:6px 0;color:#8892a4;">Probability</td>
                  <td style="color:#43e97b;font-weight:600;">True (Platt scaling)</td></tr>
              <tr><td style="padding:6px 0;color:#8892a4;">Train/Test Split</td>
                  <td style="color:#e8ecf4;font-weight:600;">80% / 20% (stratified)</td></tr>
              <tr><td style="padding:6px 0;color:#8892a4;">Feature Size</td>
                  <td style="color:#e8ecf4;font-weight:600;">4 096 (64×64 grayscale)</td></tr>
              <tr><td style="padding:6px 0;color:#8892a4;">Normalization</td>
                  <td style="color:#e8ecf4;font-weight:600;">StandardScaler</td></tr>
            </table>
          </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
          <div class="card-title">📚 Why SVM?</div>
          <div class="card-body">
            <ul style="padding-left:16px;line-height:2;">
              <li>Works well in high-dimensional spaces</li>
              <li>Effective with small-to-medium datasets</li>
              <li>Memory efficient (uses support vectors only)</li>
              <li>RBF kernel handles non-linear boundaries</li>
              <li>Probability calibration via Platt scaling</li>
            </ul>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if stats["total"] == 0:
        st.warning("⚠️  No images found in `dataset/cats/` or `dataset/dogs/`. "
                   "Please add images before training.")
    else:
        st.info(f"Ready to train on **{stats['total']} images** "
                f"({stats['cats']} cats, {stats['dogs']} dogs)")

        if st.button("🚀  Start Training", use_container_width=False):
            with st.spinner("Training in progress — this may take a minute…"):
                try:
                    import importlib, sys
                    # Dynamically import and run train_model
                    if "train_model" in sys.modules:
                        del sys.modules["train_model"]
                    import train_model as tm
                    X, y = tm.load_dataset(CATS_DIR, DOGS_DIR, MAX_IMAGES)
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y, test_size=TEST_SIZE,
                        random_state=RANDOM_STATE, stratify=y
                    )
                    X_tr_s, X_te_s, sc = tm.normalize(X_tr, X_te)
                    clf_new = tm.train_svm(X_tr_s, y_tr)
                    met = tm.evaluate(clf_new, X_te_s, y_te)
                    tm.save_artifacts(clf_new, sc, met)
                    # Clear cache so sidebar reloads
                    load_model.__wrapped__ = None  # type: ignore
                    st.success(f"✅ Training complete! Accuracy: **{met['accuracy']}%**")
                    st.balloons()
                except Exception as e:
                    st.error(f"Training failed: {e}")

    st.markdown("### 💡 Training Tips")
    tips = [
        ("More data → better accuracy",
         "Add more images to dataset/cats/ and dataset/dogs/. 1000+ per class recommended."),
        ("Tune hyperparameters",
         "Increase C for tighter fit. Try kernel='linear' if RBF overfits small datasets."),
        ("Try larger images",
         "Change IMAGE_SIZE to (128,128) in train_model.py for richer features (slower)."),
        ("Data augmentation",
         "Flip, rotate, or add noise to images to artificially expand your dataset."),
    ]
    cols = st.columns(2)
    for i, (t, d) in enumerate(tips):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="card">
              <div class="card-title">💡 {t}</div>
              <div class="card-body">{d}</div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# PAGE: EVALUATION
# ════════════════════════════════════════════════════════
elif page == "Evaluation":
    st.markdown("""
    <div class="section-head">
      <div class="section-icon" style="background:rgba(255,101,132,.15);">📈</div>
      <div>
        <div class="section-title">Model Evaluation</div>
        <div class="section-desc">Performance metrics, confusion matrix, and detailed report</div>
      </div>
    </div>""", unsafe_allow_html=True)

    metrics = load_metrics()

    if not metrics:
        st.warning("No metrics found. Train the model first (Model Training page).")
    else:
        c1, c2, c3, c4 = st.columns(4)
        metric_tiles = [
            ("Accuracy",  f"{metrics['accuracy']}%",  "#6c63ff"),
            ("Precision", f"{metrics['precision']}%", "#ff6584"),
            ("Recall",    f"{metrics['recall']}%",    "#43e97b"),
            ("F1 Score",  f"{metrics['f1_score']}%",  "#f7971e"),
        ]
        for col, (lbl, val, color) in zip([c1,c2,c3,c4], metric_tiles):
            with col:
                st.markdown(f"""
                <div class="metric-tile">
                  <div class="metric-label">{lbl}</div>
                  <div class="metric-value" style="color:{color};">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_cm, col_bar = st.columns(2)

        with col_cm:
            cm = np.array(metrics["confusion_matrix"])
            fig = chart_confusion_matrix(cm)
            st.pyplot(fig)
            plt.close(fig)

        with col_bar:
            fig = chart_metrics_radar(metrics)
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("### 📋 Classification Report")
        st.code(metrics.get("report", "Report not available"), language="text")

        st.markdown("### 📖 Metrics Explained")
        expl = [
            ("Accuracy",
             "Overall correct predictions / total predictions. "
             "Best for balanced datasets."),
            ("Precision",
             "Of all predicted Dogs, how many are actually Dogs? "
             "Important when false positives are costly."),
            ("Recall",
             "Of all actual Dogs, how many did we catch? "
             "Important when false negatives are costly."),
            ("F1 Score",
             "Harmonic mean of Precision and Recall. "
             "Best single metric for imbalanced classes."),
        ]
        cols = st.columns(2)
        for i, (t, d) in enumerate(expl):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="card">
                  <div class="card-title">{t}</div>
                  <div class="card-body">{d}</div>
                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# PAGE: PREDICTION
# ════════════════════════════════════════════════════════
elif page == "Prediction":
    st.markdown("""
    <div class="section-head">
      <div class="section-icon" style="background:rgba(56,189,248,.15);">🔍</div>
      <div>
        <div class="section-title">Image Prediction</div>
        <div class="section-desc">Upload an image to classify it as Dog or Cat</div>
      </div>
    </div>""", unsafe_allow_html=True)

    clf, scaler, img_size = load_model()

    if clf is None:
        st.warning("⚠️  Model not loaded. Please train the model first.")
    else:
        uploaded = st.file_uploader(
            "Upload a JPG / PNG / BMP image",
            type=["jpg","jpeg","png","bmp","webp"],
            label_visibility="visible"
        )

        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            col_img, col_res = st.columns([1, 1.2])

            with col_img:
                st.markdown("**Uploaded Image**")
                st.image(pil_img, use_column_width=True)
                st.caption(f"Size: {pil_img.size[0]} × {pil_img.size[1]} px")

            with col_res:
                if st.button("🔮  Classify Image", use_container_width=True):
                    with st.spinner("Analysing…"):
                        img_arr  = np.array(pil_img)
                        features = preprocess_image(img_arr, tuple(img_size))
                        features_s = scaler.transform([features])
                        pred     = clf.predict(features_s)[0]
                        proba    = clf.predict_proba(features_s)[0]
                        conf     = proba[pred] * 100
                        alt_conf = proba[1 - pred] * 100

                    cls      = "cat" if pred == 0 else "dog"
                    emoji    = "🐱" if pred == 0 else "🐶"
                    label    = "Cat" if pred == 0 else "Dog"
                    color    = "#ff6584" if pred == 0 else "#6c63ff"

                    st.markdown(f"""
                    <div class="pred-box {cls}">
                      <div class="pred-animal">{emoji}</div>
                      <div class="pred-label">It's a {label}!</div>
                      <div class="pred-conf">Confidence: <b style='color:{color}'>{conf:.1f}%</b></div>
                      <div class="conf-bar-wrap">
                        <div class="conf-bar" style="width:{conf}%;background:{color};"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**Probability Breakdown**")
                    prob_df = pd.DataFrame({
                        "Class":       ["🐱 Cat", "🐶 Dog"],
                        "Probability": [f"{proba[0]*100:.2f}%", f"{proba[1]*100:.2f}%"],
                        "Confidence":  [proba[0]*100, proba[1]*100],
                    })
                    st.dataframe(prob_df[["Class","Probability"]],
                                 use_container_width=True)

                    # Downloadable result
                    result_str = (
                        f"Prediction: {label}\n"
                        f"Confidence: {conf:.2f}%\n"
                        f"Cat Prob: {proba[0]*100:.2f}%\n"
                        f"Dog Prob: {proba[1]*100:.2f}%\n"
                        f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
                    st.download_button(
                        "⬇️  Download Result",
                        data=result_str,
                        file_name=f"prediction_{label.lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )


# ════════════════════════════════════════════════════════
# PAGE: VISUALIZATIONS
# ════════════════════════════════════════════════════════
elif page == "Visualizations":
    st.markdown("""
    <div class="section-head">
      <div class="section-icon" style="background:rgba(167,139,250,.15);">📉</div>
      <div>
        <div class="section-title">Visualizations</div>
        <div class="section-desc">Charts, graphs, and visual model insights</div>
      </div>
    </div>""", unsafe_allow_html=True)

    stats   = load_dataset_stats()
    metrics = load_metrics()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Class Distribution", "📈 Metrics Comparison",
         "🗺️ Confusion Matrix", "⚙️ Preprocessing Pipeline"]
    )

    with tab1:
        if stats["total"] > 0:
            col1, col2 = st.columns(2)
            with col1:
                fig = chart_class_distribution(stats)
                st.pyplot(fig); plt.close(fig)
            with col2:
                # Pie chart
                fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor=BG)
                ax2.set_facecolor(BG)
                wedges, texts, autotexts = ax2.pie(
                    [stats["cats"], stats["dogs"]],
                    labels=["Cats", "Dogs"],
                    colors=[accent("pink"), accent("purple")],
                    autopct="%1.1f%%", startangle=140,
                    textprops={"color": TXT, "fontsize": 11},
                    wedgeprops={"edgecolor": BG, "linewidth": 2}
                )
                for at in autotexts:
                    at.set_color(TXT); at.set_fontsize(10)
                ax2.set_title("Class Share", color=TXT, fontsize=12, fontweight="bold")
                st.pyplot(fig2); plt.close(fig2)
        else:
            st.info("No dataset images found.")

    with tab2:
        if metrics:
            fig = chart_metrics_radar(metrics)
            st.pyplot(fig); plt.close(fig)
        else:
            st.info("Train the model to see metrics.")

    with tab3:
        if metrics:
            cm  = np.array(metrics["confusion_matrix"])
            fig = chart_confusion_matrix(cm)
            st.pyplot(fig); plt.close(fig)

            # Normalised CM
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fig2    = chart_confusion_matrix(
                (cm_norm * 100).astype(int), "Normalised Confusion Matrix (%)"
            )
            st.pyplot(fig2); plt.close(fig2)
        else:
            st.info("Train the model to see confusion matrix.")

    with tab4:
        fig = chart_preprocessing_pipeline()
        st.pyplot(fig); plt.close(fig)

        # Pixel intensity distribution if dataset exists
        if stats["cats"] > 0:
            cat_files = [f for f in os.listdir(CATS_DIR)
                         if os.path.splitext(f)[1].lower() in VALID_EXTS][:50]
            pixels = []
            for f in cat_files:
                img = cv2.imread(os.path.join(CATS_DIR, f), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    pixels.extend(img.flatten().tolist())
            if pixels:
                fig3, ax3 = _dark_fig(7, 3)
                ax3.hist(pixels, bins=64, color=accent("pink"), alpha=.75, edgecolor="none")
                ax3.set_title("Pixel Intensity Distribution (Cat samples)",
                              fontsize=11, fontweight="bold")
                ax3.set_xlabel("Pixel Value (0-255)")
                ax3.set_ylabel("Frequency")
                ax3.grid(axis="y", color=BDR, linewidth=.5)
                st.pyplot(fig3); plt.close(fig3)


# ════════════════════════════════════════════════════════
# PAGE: DOWNLOADS
# ════════════════════════════════════════════════════════
elif page == "Downloads":
    st.markdown("""
    <div class="section-head">
      <div class="section-icon" style="background:rgba(247,151,30,.15);">⬇️</div>
      <div>
        <div class="section-title">Downloads</div>
        <div class="section-desc">Export your trained model, metrics, and reports</div>
      </div>
    </div>""", unsafe_allow_html=True)

    clf, scaler, img_size = load_model()
    metrics = load_metrics()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
          <div class="card-title">🧠 Trained Model (model.pkl)</div>
          <div class="card-body">Download the pickled SVM model bundle including the trained
          classifier and fitted StandardScaler. Load with <code>pickle.load()</code>.</div>
        </div>""", unsafe_allow_html=True)
        if clf and os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                st.download_button("⬇️  Download model.pkl", f.read(),
                                   file_name="model.pkl",
                                   mime="application/octet-stream",
                                   use_container_width=True)
        else:
            st.info("No model found. Train the model first.")

    with col2:
        st.markdown("""
        <div class="card">
          <div class="card-title">📊 Metrics Report (metrics.json)</div>
          <div class="card-body">Download the JSON evaluation report containing accuracy,
          precision, recall, F1 score, and the full confusion matrix.</div>
        </div>""", unsafe_allow_html=True)
        if metrics and os.path.exists(METRICS_PATH):
            with open(METRICS_PATH) as f:
                content = f.read()
            st.download_button("⬇️  Download metrics.json", content,
                               file_name="metrics.json", mime="application/json",
                               use_container_width=True)
        else:
            st.info("No metrics found. Train the model first.")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <div class="card-title">📖 How to use the downloaded model</div>
    </div>""", unsafe_allow_html=True)
    st.code("""import pickle
import cv2
import numpy as np

# Load the bundle
with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)

clf     = bundle["model"]
scaler  = bundle["scaler"]
img_size = bundle["image_size"]   # (64, 64)

def predict(image_path):
    img   = cv2.imread(image_path)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res   = cv2.resize(gray, img_size)
    flat  = res.flatten().astype(np.float32).reshape(1, -1)
    scaled = scaler.transform(flat)
    pred  = clf.predict(scaled)[0]
    proba = clf.predict_proba(scaled)[0]
    label = "Cat" if pred == 0 else "Dog"
    conf  = proba[pred] * 100
    return label, conf

label, conf = predict("my_pet.jpg")
print(f"{label} ({conf:.1f}% confidence)")""", language="python")

    if metrics:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("### 📋 Current Model Summary")
        summary = {
            "Accuracy (%)":  metrics["accuracy"],
            "Precision (%)": metrics["precision"],
            "Recall (%)":    metrics["recall"],
            "F1 Score (%)":  metrics["f1_score"],
            "Test Samples":  metrics.get("test_size", "N/A"),
            "Image Size":    "×".join(str(x) for x in metrics.get("image_size", [64,64])),
        }
        df = pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False)
        st.download_button("⬇️  Download Summary CSV", csv,
                           file_name="model_summary.csv", mime="text/csv")
