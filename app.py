"""
Dog vs Cat Image Classification — Streamlit Dashboard
======================================================
Run:  streamlit run app.py
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import subprocess
import sys

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Dog vs Cat Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ───────────────────────────────────────────────────────────────
IMG_SIZE    = (64, 64)
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "svm_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LABELS_PATH = os.path.join(MODEL_DIR, "label_map.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.pkl")
DATASET_DIR  = "dataset"

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Google Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* ── Background ── */
  .stApp {
    background: linear-gradient(135deg, #0d0f1a 0%, #111827 60%, #0d1117 100%);
    color: #e2e8f0;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0d1117 100%) !important;
    border-right: 1px solid #1e293b;
  }
  section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

  /* ── Hero banner ── */
  .hero {
    background: linear-gradient(120deg, #1e3a5f 0%, #162032 50%, #1a1040 100%);
    border: 1px solid #2d4a6e;
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(circle at 70% 50%, rgba(56,189,248,0.08) 0%, transparent 70%);
  }
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
    line-height: 1.1;
  }
  .hero-sub {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 300;
    margin: 0;
  }
  .badge {
    display: inline-block;
    background: rgba(56,189,248,0.15);
    border: 1px solid rgba(56,189,248,0.35);
    color: #38bdf8;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
  }

  /* ── Metric cards ── */
  .metric-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.5rem; }
  .metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: transform .2s, border-color .2s;
  }
  .metric-card:hover { transform: translateY(-3px); border-color: #38bdf8; }
  .metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #38bdf8;
  }
  .metric-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: .06em; }

  /* ── Section headers ── */
  .section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #e2e8f0;
    border-left: 3px solid #38bdf8;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
  }

  /* ── Prediction result ── */
  .pred-box {
    background: linear-gradient(135deg, #0f2942, #0a1929);
    border: 2px solid #38bdf8;
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
  }
  .pred-animal {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .pred-conf {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: 0.25rem;
  }
  .conf-bar-wrap { background: #1e293b; border-radius: 999px; height: 10px; margin: 1rem 0 0.3rem; overflow: hidden; }
  .conf-bar { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #38bdf8, #818cf8); }

  /* ── Info / warning boxes ── */
  .info-box {
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    color: #cbd5e1;
    font-size: 0.9rem;
    line-height: 1.6;
  }
  .warn-box {
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    color: #fde68a;
    font-size: 0.9rem;
  }

  /* ── Streamlit widgets overrides ── */
  .stButton > button {
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    color: #0d0f1a !important;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.8rem;
    font-size: 0.95rem;
    transition: opacity .2s, transform .15s;
  }
  .stButton > button:hover { opacity: 0.88; transform: translateY(-2px); }
  div[data-testid="stFileUploader"] {
    background: #1e293b;
    border: 2px dashed #334155;
    border-radius: 14px;
    padding: 1rem;
  }
  .stSelectbox select, .stSelectbox > div > div {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border-color: #334155 !important;
  }
  .stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 10px; padding: 4px; }
  .stTabs [data-baseweb="tab"] { color: #64748b !important; border-radius: 8px; }
  .stTabs [aria-selected="true"] { background: #1e293b !important; color: #38bdf8 !important; }
  .stProgress > div > div > div { background: linear-gradient(90deg, #38bdf8, #818cf8); }
  h1, h2, h3, h4 { color: #e2e8f0 !important; }
  p, li, label, span { color: #cbd5e1; }
  .stMarkdown p { color: #cbd5e1; }
</style>
""", unsafe_allow_html=True)


# ─── Utility functions ────────────────────────────────────────────────────────

@st.cache_resource
def load_model_artifacts():
    """Load SVM model, scaler, label map, and metrics from disk."""
    try:
        with open(MODEL_PATH,   "rb") as f: model     = pickle.load(f)
        with open(SCALER_PATH,  "rb") as f: scaler    = pickle.load(f)
        with open(LABELS_PATH,  "rb") as f: label_map = pickle.load(f)
        with open(METRICS_PATH, "rb") as f: metrics   = pickle.load(f)
        return model, scaler, label_map, metrics, True
    except FileNotFoundError:
        return None, None, None, None, False


def preprocess_for_prediction(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.flatten().reshape(1, -1)


def dark_fig():
    """Return a pre-configured dark-theme Figure + Axes."""
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#1e293b")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.tick_params(colors="#94a3b8")
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_color("#94a3b8")
    ax.title.set_color("#e2e8f0")
    return fig, ax


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
      <div style='font-size:3rem;'>🐾</div>
      <div style='font-family:Syne,sans-serif; font-size:1.1rem; font-weight:700; color:#e2e8f0;'>Dog vs Cat SVM</div>
      <div style='font-size:0.75rem; color:#64748b; margin-top:4px;'>Image Classification Engine</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔬 Predict", "📊 Evaluation", "📖 About"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    model, scaler, label_map, metrics, model_loaded = load_model_artifacts()

    if model_loaded:
        st.markdown('<div class="section-header">Model Status</div>', unsafe_allow_html=True)
        st.success("✅ Model ready")
        if metrics:
            st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
            st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
    else:
        st.markdown('<div class="warn-box">⚠️ Model not trained yet.<br>Click <b>Train Model</b> on the Dashboard.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem; color:#475569; text-align:center;">Built with Streamlit + scikit-learn<br>SVM · RBF Kernel · Python 3.10+</p>', unsafe_allow_html=True)


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="badge">🤖 Machine Learning · SVM Classifier</div>
  <div class="hero-title">Dog vs Cat<br>Image Classifier</div>
  <p class="hero-sub">Support Vector Machine · RBF Kernel · scikit-learn · Streamlit</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    # ── Top metrics row ──────────────────────────────────────────
    if model_loaded and metrics:
        st.markdown(f"""
        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-value">{metrics['total_samples']}</div>
            <div class="metric-label">Total Samples</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">{metrics['dogs_count']}</div>
            <div class="metric-label">Dog Images</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">{metrics['cats_count']}</div>
            <div class="metric-label">Cat Images</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">{metrics['accuracy']*100:.1f}%</div>
            <div class="metric-label">Test Accuracy</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-grid">
          <div class="metric-card"><div class="metric-value">—</div><div class="metric-label">Total Samples</div></div>
          <div class="metric-card"><div class="metric-value">—</div><div class="metric-label">Dog Images</div></div>
          <div class="metric-card"><div class="metric-value">—</div><div class="metric-label">Cat Images</div></div>
          <div class="metric-card"><div class="metric-value">—</div><div class="metric-label">Test Accuracy</div></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Train button ──────────────────────────────────────────────
    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown('<div class="section-header">Model Training</div>', unsafe_allow_html=True)
        if st.button("🚀 Train Model"):
            with st.spinner("Training SVM model … this may take a moment"):
                result = subprocess.run(
                    [sys.executable, "train_model.py"],
                    capture_output=True, text=True
                )
            if result.returncode == 0:
                st.success("✅ Model trained and saved successfully!")
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error("Training failed. See details below.")
                st.code(result.stderr or result.stdout)

        st.markdown("""
        <div class="info-box">
          <b>What happens when you train?</b><br>
          A synthetic dog/cat dataset is generated, split 80/20, scaled with
          StandardScaler, then an RBF-SVM is fitted and saved to <code>models/</code>.
          Real Kaggle data can be placed in <code>dataset/dogs/</code> and
          <code>dataset/cats/</code> for production-level accuracy.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Pipeline Overview</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 2.6))
        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#111827")
        ax.axis("off")

        steps = ["Load\nImages", "Resize\n64×64", "Normalise\n÷255", "StandardScaler", "SVM\nRBF", "Predict"]
        colors = ["#1e3a5f","#1e3a5f","#1e3a5f","#1a1040","#0f2942","#1a1040"]
        text_c = ["#38bdf8","#38bdf8","#38bdf8","#818cf8","#f472b6","#34d399"]

        for i, (step, col, tc) in enumerate(zip(steps, colors, text_c)):
            x = i / (len(steps) - 1)
            ax.add_patch(mpatches.FancyBboxPatch(
                (x - 0.07, 0.25), 0.14, 0.5,
                boxstyle="round,pad=0.02",
                facecolor=col, edgecolor=tc, linewidth=1.5,
            ))
            ax.text(x, 0.5, step, ha="center", va="center",
                    color=tc, fontsize=7.5, fontweight="bold")
            if i < len(steps) - 1:
                ax.annotate("", xy=(x + 0.09, 0.5), xytext=(x + 0.07, 0.5),
                            arrowprops=dict(arrowstyle="->", color="#475569", lw=1.5))

        ax.set_xlim(-0.1, 1.1); ax.set_ylim(0, 1)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Dataset preview ──────────────────────────────────────────
    st.markdown('<div class="section-header">Sample Dataset Preview</div>', unsafe_allow_html=True)
    cats_dir = os.path.join(DATASET_DIR, "cats")
    dogs_dir = os.path.join(DATASET_DIR, "dogs")

    if os.path.isdir(cats_dir) and os.path.isdir(dogs_dir):
        cat_imgs = sorted([f for f in os.listdir(cats_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])[:4]
        dog_imgs = sorted([f for f in os.listdir(dogs_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])[:4]
        cols = st.columns(8)
        for idx, fname in enumerate(cat_imgs):
            with cols[idx]:
                img = Image.open(os.path.join(cats_dir, fname))
                st.image(img, caption="🐱 Cat", use_container_width=True)
        for idx, fname in enumerate(dog_imgs):
            with cols[4 + idx]:
                img = Image.open(os.path.join(dogs_dir, fname))
                st.image(img, caption="🐶 Dog", use_container_width=True)
    else:
        st.markdown('<div class="warn-box">Dataset not found. Train the model first to generate sample data.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ════════════════════════════════════════════════════════════════════
elif page == "🔬 Predict":
    st.markdown('<div class="section-header">Upload an Image</div>', unsafe_allow_html=True)

    if not model_loaded:
        st.markdown('<div class="warn-box">⚠️ Please train the model first from the Dashboard.</div>', unsafe_allow_html=True)
    else:
        uploaded = st.file_uploader(
            "Drop a dog or cat image here",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded:
            img = Image.open(uploaded)
            col_img, col_res = st.columns([1, 1])

            with col_img:
                st.markdown('<div class="section-header">Uploaded Image</div>', unsafe_allow_html=True)
                st.image(img, use_container_width=True)
                w, h = img.size
                st.caption(f"Original: {w}×{h}px  ·  Mode: {img.mode}")

            with col_res:
                st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)
                with st.spinner("Analysing …"):
                    feat   = preprocess_for_prediction(img)
                    feat_s = scaler.transform(feat)
                    pred   = model.predict(feat_s)[0]
                    proba  = model.predict_proba(feat_s)[0]
                    conf   = float(proba[pred])
                    label  = label_map[pred]
                    emoji  = "🐶" if label == "Dog" else "🐱"
                    alt    = "Cat" if label == "Dog" else "Dog"
                    alt_conf = float(proba[1 - pred])

                bar_pct = int(conf * 100)
                st.markdown(f"""
                <div class="pred-box">
                  <div style="font-size:3.5rem;">{emoji}</div>
                  <div class="pred-animal">{label}</div>
                  <div class="pred-conf">Confidence: {conf*100:.1f}%</div>
                  <div class="conf-bar-wrap">
                    <div class="conf-bar" style="width:{bar_pct}%;"></div>
                  </div>
                  <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#475569;">
                    <span>0%</span><span>50%</span><span>100%</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Probability bar chart
                fig, ax = dark_fig()
                fig.set_size_inches(4, 2.2)
                labels_b = ["Cat 🐱", "Dog 🐶"]
                vals     = [float(proba[0]), float(proba[1])]
                clrs     = ["#818cf8", "#38bdf8"]
                bars = ax.barh(labels_b, vals, color=clrs, height=0.45)
                for bar, v in zip(bars, vals):
                    ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                            f"{v*100:.1f}%", va="center", color="#e2e8f0", fontsize=10)
                ax.set_xlim(0, 1.15)
                ax.set_xlabel("Probability")
                ax.set_title("Class Probabilities", pad=8, fontsize=10)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                st.markdown(f"""
                <div class="info-box" style="margin-top:1rem;">
                  <b>Interpretation:</b><br>
                  The model is <b>{conf*100:.1f}% confident</b> this is a <b>{label}</b>.
                  Alternative: {alt} ({alt_conf*100:.1f}%)
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box" style="text-align:center; padding:2rem;">
              <div style="font-size:3rem; margin-bottom:0.5rem;">📸</div>
              Upload a <b>.jpg</b> or <b>.png</b> image of a dog or cat above to get a prediction.
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: EVALUATION
# ════════════════════════════════════════════════════════════════════
elif page == "📊 Evaluation":
    st.markdown('<div class="section-header">Model Evaluation Metrics</div>', unsafe_allow_html=True)

    if not model_loaded or metrics is None:
        st.markdown('<div class="warn-box">⚠️ Train the model first to see evaluation metrics.</div>', unsafe_allow_html=True)
    else:
        # ── Metrics cards ──────────────────────────────────────
        st.markdown(f"""
        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-value">{metrics['accuracy']*100:.1f}%</div>
            <div class="metric-label">Accuracy</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">{metrics['precision']:.4f}</div>
            <div class="metric-label">Precision</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">{metrics['recall']:.4f}</div>
            <div class="metric-label">Recall</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">{metrics['f1_score']:.4f}</div>
            <div class="metric-label">F1 Score</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_cm, col_bar = st.columns(2)

        # ── Confusion Matrix ────────────────────────────────────
        with col_cm:
            st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
            cm = metrics["confusion_matrix"]
            fig, ax = plt.subplots(figsize=(4.5, 3.8))
            fig.patch.set_facecolor("#111827")
            ax.set_facecolor("#1e293b")
            cmap = sns.color_palette(["#0f172a","#0f2942","#1e3a5f","#1d4ed8","#38bdf8"], as_cmap=True)
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Cat","Dog"], yticklabels=["Cat","Dog"],
                ax=ax, linewidths=1, linecolor="#334155",
                annot_kws={"size":14,"color":"white","weight":"bold"},
            )
            ax.set_xlabel("Predicted", color="#94a3b8")
            ax.set_ylabel("Actual",    color="#94a3b8")
            ax.set_title("Confusion Matrix", color="#e2e8f0", pad=10)
            ax.tick_params(colors="#94a3b8")
            plt.setp(ax.get_xticklabels(), color="#94a3b8")
            plt.setp(ax.get_yticklabels(), color="#94a3b8")
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_tick_params(color="#94a3b8")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#94a3b8")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # ── Metrics bar chart ───────────────────────────────────
        with col_bar:
            st.markdown('<div class="section-header">Metrics Comparison</div>', unsafe_allow_html=True)
            metric_names = ["Accuracy","Precision","Recall","F1 Score"]
            metric_vals  = [
                metrics["accuracy"], metrics["precision"],
                metrics["recall"],   metrics["f1_score"],
            ]
            clrs = ["#38bdf8","#818cf8","#f472b6","#34d399"]
            fig, ax = dark_fig()
            fig.set_size_inches(4.5, 3.8)
            bars = ax.bar(metric_names, metric_vals, color=clrs, width=0.5)
            for bar, v in zip(bars, metric_vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                        f"{v:.4f}", ha="center", va="bottom",
                        color="#e2e8f0", fontsize=9, fontweight="bold")
            ax.set_ylim(0, 1.12)
            ax.set_title("Evaluation Metrics", pad=10)
            ax.set_ylabel("Score")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # ── Train / Test distribution ────────────────────────────
        st.markdown('<div class="section-header">Train / Test Split</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            fig, ax = dark_fig()
            fig.set_size_inches(4, 3)
            sizes  = [metrics["train_size"], metrics["test_size"]]
            labels = ["Train", "Test"]
            explo  = [0.04, 0.04]
            colors = ["#38bdf8","#818cf8"]
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, autopct="%1.1f%%",
                explode=explo, colors=colors,
                textprops={"color":"#e2e8f0","fontsize":11},
                wedgeprops={"linewidth":1.5,"edgecolor":"#0d0f1a"},
            )
            for at in autotexts: at.set_color("#0d0f1a"); at.set_fontweight("bold")
            ax.set_title("Data Split", color="#e2e8f0", pad=12)
            fig.patch.set_facecolor("#111827")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_b:
            fig, ax = dark_fig()
            fig.set_size_inches(4, 3)
            ax.bar(["Cats","Dogs"],
                   [metrics["cats_count"], metrics["dogs_count"]],
                   color=["#818cf8","#38bdf8"], width=0.45)
            ax.set_title("Class Distribution", pad=10)
            ax.set_ylabel("Count")
            for i, v in enumerate([metrics["cats_count"], metrics["dogs_count"]]):
                ax.text(i, v + 2, str(v), ha="center", color="#e2e8f0", fontweight="bold")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ════════════════════════════════════════════════════════════════════
elif page == "📖 About":
    st.markdown('<div class="section-header">About this Project</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🤖 SVM Algorithm", "🖼 Image Preprocessing", "🗂 Project Structure"])

    with tab1:
        st.markdown("""
        <div class="info-box">
        <h4 style="color:#38bdf8; margin-top:0;">Support Vector Machine (SVM)</h4>

        SVM is a powerful supervised learning algorithm used for classification and regression.
        It finds the optimal <b>hyperplane</b> that maximises the margin between classes.

        <h5 style="color:#818cf8;">Key Concepts</h5>
        <ul>
          <li><b>Support Vectors</b> — the data points closest to the decision boundary.</li>
          <li><b>Margin</b> — the gap between the hyperplane and the nearest support vectors; SVM maximises this.</li>
          <li><b>RBF Kernel</b> — maps data into a higher-dimensional space to handle non-linear boundaries.</li>
          <li><b>C Parameter</b> — trades off margin width vs. classification errors (C=10 used here).</li>
        </ul>

        <h5 style="color:#818cf8;">Why SVM for Images?</h5>
        <ul>
          <li>Effective in high-dimensional feature spaces (64×64×3 = 12,288 dims).</li>
          <li>Memory-efficient — only support vectors are stored, not the full dataset.</li>
          <li>Robust to overfitting with the right regularisation parameter.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class="info-box">
        <h4 style="color:#38bdf8; margin-top:0;">Image Preprocessing Pipeline</h4>

        <ol>
          <li><b>Load image</b> — PIL opens any JPEG/PNG.</li>
          <li><b>Convert to RGB</b> — ensures 3-channel consistency regardless of source format.</li>
          <li><b>Resize to 64×64</b> — reduces dimensionality while retaining enough structure.</li>
          <li><b>Normalise (÷255)</b> — scales pixel values from [0,255] to [0,1] for numerical stability.</li>
          <li><b>Flatten</b> — converts the 64×64×3 array to a 12,288-element 1-D feature vector.</li>
          <li><b>StandardScaler</b> — zero-centres and unit-variances each feature; critical for SVM's distance-based kernel.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div class="info-box">
        <h4 style="color:#38bdf8; margin-top:0;">Folder Structure</h4>
        <pre style="color:#94a3b8; background:#0d1117; padding:1rem; border-radius:8px; font-size:0.85rem;">
dog_vs_cat_svm/
├── app.py               ← Streamlit dashboard
├── train_model.py       ← Data loading + SVM training
├── requirements.txt     ← Python dependencies
├── README.md            ← Full documentation
├── models/
│   ├── svm_model.pkl    ← Trained SVM (generated)
│   ├── scaler.pkl       ← StandardScaler (generated)
│   ├── label_map.pkl    ← {0:"Cat", 1:"Dog"}
│   └── metrics.pkl      ← Evaluation results
└── dataset/
    ├── dogs/            ← Dog images (.jpg)
    └── cats/            ← Cat images (.jpg)
        </pre>
        <b>To use real Kaggle data:</b>
        Download the <i>Dogs vs. Cats</i> dataset from Kaggle, extract, and copy images into
        <code>dataset/dogs/</code> and <code>dataset/cats/</code>, then re-train.
        </div>
        """, unsafe_allow_html=True)