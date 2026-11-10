import streamlit as st
import os
import tempfile
import sys
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Load the ensemble Detector (cached so weights stay in memory) ──────────────
@st.cache_resource(show_spinner="Loading model weights… (first run may take a minute)")
def load_detector():
    """Load the dfd_lib Detector from the bundled .pkl file."""
    lib_dir = Path(__file__).parent
    sys.path.insert(0, str(lib_dir))
    import dfd_lib as L  # noqa: F401

    pkl_path = lib_dir / "deepfake_detector.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {pkl_path}\n"
            "Place deepfake_detector.pkl in the same directory as Streamlit.py."
        )
    detector = L.Detector(str(pkl_path))
    return detector


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    n_frames = st.slider(
        "Frames to sample per video",
        min_value=4,
        max_value=32,
        value=16,
        step=4,
        help="More frames = more accurate but slower. 16 is a good default.",
    )
    st.divider()
    st.markdown(
        """
**Model info**
- **Architecture:** Ensemble — CLIP ViT-L/14 (LayerNorm + SVD adapters) + EfficientNet-B4
- **Trained on:** FaceForensics++ c23 + Celeb-DF v2
- **Ensemble AUC:** 1.00 (CelebDF), 0.989 (FF++)
- **Input:** MP4 / AVI / MOV / MKV / WEBM video or JPG / PNG image
        """
    )
    st.divider()
    with st.expander("📊 Final metrics"):
        try:
            import pandas as pd
            metrics_path = Path(__file__).parent / "final_metrics.csv"
            if metrics_path.exists():
                df = pd.read_csv(metrics_path)
                cols = ["set", "AUC", "AP", "accuracy", "F1", "MCC"]
                show_cols = [c for c in cols if c in df.columns]
                st.dataframe(
                    df[show_cols].set_index("set").round(4),
                    use_container_width=True,
                )
        except Exception as e:
            st.warning(f"Could not load metrics: {e}")


# ── Main UI ────────────────────────────────────────────────────────────────────
st.title("🕵️‍♂️ Deepfake Detection App")
st.markdown(
    "Upload a video (or a face image) and the ensemble model — "
    "**CLIP ViT-L/14 + EfficientNet-B4** trained on FF++ & Celeb-DF v2 — "
    "will return a calibrated **fake probability** with per-model scores."
)

uploaded_file = st.file_uploader(
    "Upload a Video or Image",
    type=["mp4", "avi", "mov", "mkv", "webm", "jpg", "jpeg", "png"],
    help="Short clips (< 1 min) work best. Longer videos are sampled evenly.",
)

if uploaded_file is not None:
    is_video = uploaded_file.type.startswith("video")
    col_media, col_res = st.columns([1, 1])

    with col_media:
        if is_video:
            st.video(uploaded_file)
        else:
            st.image(uploaded_file, use_container_width=True)

    with col_res:
        frame_label = f"{n_frames} frames" if is_video else "face crop"
        with st.spinner(f"Running deepfake detection ({frame_label})…"):
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                detector = load_detector()
                result = detector.predict(tmp_path, n_frames=n_frames)
            except Exception as exc:
                st.error(f"❌ Detection failed: {exc}")
                result = None
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        if result is None:
            pass  # error already shown above
        elif "error" in result:
            st.warning(f"⚠️ {result['error']}")
            st.info("Tip: make sure the input contains a clearly visible face.")
        else:
            label = result["label"]          # "FAKE" or "REAL"
            prob_fake = result["prob_fake"]  # calibrated probability [0, 1]
            score = result["score"]          # raw ensemble logit

            # ── Verdict banner ──
            if label == "FAKE":
                st.error(f"🚨 **FAKE** — {prob_fake:.1%} fake probability")
            else:
                st.success(f"✅ **REAL** — {1 - prob_fake:.1%} real probability")

            # ── Confidence gauge ──
            st.markdown("**Fake probability**")
            st.progress(prob_fake)

            # ── Detailed breakdown ──
            with st.expander("🔍 Detailed scores"):
                crops = result.get("crops", [])
                n_crops = len(crops) if isinstance(crops, list) else "?"
                st.markdown(f"- **Raw ensemble score:** `{score:+.4f}` (positive → FAKE)")
                st.markdown(f"- **Faces analysed:** `{n_crops}` frames")

                if "per_model" in result and result["per_model"]:
                    st.markdown("**Per-model logits (before ensemble weighting):**")
                    for model_name, logit in result["per_model"].items():
                        st.markdown(f"  - `{model_name}`: `{logit:+.4f}`")

                if "frame_scores" in result and result["frame_scores"]:
                    import pandas as pd
                    fs = result["frame_scores"]
                    st.markdown("**Per-frame ensemble scores:**")
                    chart_data = pd.DataFrame(
                        {"frame": list(range(len(fs))), "score": fs}
                    ).set_index("frame")
                    st.line_chart(chart_data)
