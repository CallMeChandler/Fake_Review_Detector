import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️‍♂️", layout="centered")

BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
PIPE_PATH  = MODELS_DIR / "text_svm_pipeline.joblib"

@st.cache_resource
def load_pipe(path_str, cache_bust):
    pipe = joblib.load(path_str)
    tfidf = pipe.named_steps.get("tfidf", None)
    if tfidf is None or not hasattr(tfidf, "idf_"):
        raise RuntimeError("Loaded pipeline's TF-IDF is not fitted. Re-save the pipeline AFTER fitting (pipe.fit(...)).")
    return pipe

def predict_margin(texts, pipe):
    m = pipe.decision_function(texts)
    if isinstance(m, list):
        m = np.asarray(m)
    classes = pipe.named_steps["clf"].classes_
    if m.ndim == 1:
        margins = m
        pos_label, neg_label = classes[1], classes[0]
        return margins, pos_label, neg_label, classes
    else:
        margins = m.max(axis=1)
        return margins, classes[1], classes[0], classes

def pretty_label(lbl, pos_label, neg_label):
    if isinstance(lbl, str):
        u = lbl.upper()
        if u == "OR":
            return "Fake review"
        if u == "CG":
            return "Original review"
    if lbl == pos_label:
        return "Fake review"
    if lbl == neg_label:
        return "Original review"
    return str(lbl)

# ---------- UI ----------
st.title("🛒 Fake Product Review Detector")
st.write("Paste a review and I’ll predict whether it’s a **Fake review** or an **Original review** using TF-IDF + Linear SVM.")

if not PIPE_PATH.exists():
    st.error(f"Model file not found: {PIPE_PATH}")
    st.stop()

try:
    cache_bust = PIPE_PATH.stat().st_mtime
    pipe = load_pipe(str(PIPE_PATH), cache_bust)
except Exception as e:
    st.error(f"Could not load pipeline from {PIPE_PATH}.\n\n{e}")
    st.stop()

classes = pipe.named_steps["clf"].classes_
if len(classes) != 2:
    st.warning(f"Expected binary classes, found: {classes}")

# Sidebar
st.sidebar.header("Inference Settings")
th = st.sidebar.slider(
    "Decision threshold (margin)", -2.0, 2.0, 0.0, 0.01,
    help="> 0 favors calling a review **Fake**. Move right to be stricter about flagging a review as Fake."
)
show_margin = st.sidebar.checkbox("Show raw margin", value=True)
st.sidebar.markdown("---")
st.sidebar.write("**Batch mode** available below the text box.")

# ---------- NEW: persist outputs in session_state ----------
if "single_pred_text" not in st.session_state:
    st.session_state.single_pred_text = None
if "single_margin" not in st.session_state:
    st.session_state.single_margin = None

# ---------- SINGLE PREDICTION (form to avoid flicker/blank) ----------
with st.form("single_review_form", clear_on_submit=False):
    txt = st.text_area("Enter a review", height=140, placeholder="Type/paste a product review here…", key="single_text")
    submitted = st.form_submit_button("Predict")
    if submitted:
        if not txt or not txt.strip():
            st.warning("Please paste a review.")
        else:
            margins, pos_label, neg_label, classes = predict_margin([txt], pipe)
            margin = float(margins[0])
            pred = pos_label if margin >= th else neg_label
            st.session_state.single_pred_text = pretty_label(pred, pos_label, neg_label)
            st.session_state.single_margin = margin

# Show last prediction persistently (even after rerun)
if st.session_state.single_pred_text is not None:
    st.subheader(f"Prediction: **{st.session_state.single_pred_text}**")
    if show_margin and st.session_state.single_margin is not None:
        st.caption(f"Margin: {st.session_state.single_margin:.4f}  |  Threshold: {th:.2f}")

st.markdown("---")
st.subheader("📦 Batch predictions (CSV)")

# ---------- BATCH (form to prevent accidental reruns) ----------
with st.form("batch_form", clear_on_submit=False):
    uploaded = st.file_uploader("Upload CSV", type=["csv"], help="CSV with a text column (e.g., clean_text or text_).", key="batch_uploader")
    run_batch = st.form_submit_button("Run batch")
    if run_batch:
        if uploaded is None:
            st.warning("Please upload a CSV first.")
        else:
            try:
                df = pd.read_csv(uploaded)
                text_cols = [c for c in df.columns if df[c].dtype == object]
                if not text_cols:
                    st.error("No text-like columns found.")
                else:
                    col = st.selectbox("Select text column", text_cols, key="batch_text_col")
                    # Important: selectbox creates a rerun; guard with form so we only compute on submit
                    texts = df[col].astype(str).tolist()
                    margins, pos_label, neg_label, classes = predict_margin(texts, pipe)
                    preds = [pos_label if m >= th else neg_label for m in margins]
                    preds_text = [pretty_label(p, pos_label, neg_label) for p in preds]

                    out = df.copy()
                    out["prediction"] = preds
                    out["prediction_text"] = preds_text
                    out["margin"] = margins

                    st.success("Done. Sample:")
                    st.dataframe(out.head(10))
                    csv = out.to_csv(index=False).encode("utf-8")
                    st.download_button("Download results CSV", data=csv, file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Could not process CSV: {e}")
