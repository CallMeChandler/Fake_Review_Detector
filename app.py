import streamlit as st
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline as SKPipeline

st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️‍♂️", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
PIPE_PATH = MODELS_DIR / "text_svm_pipeline_new.joblib"
FEATS_PATH = MODELS_DIR / "feats_transformer.joblib"
CLF_PATH = MODELS_DIR / "linear_svc_model.joblib"
THR_PATH = MODELS_DIR / "threshold.json"

@st.cache_resource
def load_pipe():
    if PIPE_PATH.exists():
        obj = joblib.load(PIPE_PATH)
        if not hasattr(obj, "named_steps") or "clf" not in obj.named_steps:
            raise RuntimeError("Loaded object is not a sklearn Pipeline with a 'clf' step.")
        try:
            _ = obj.decision_function(["hello world"])
        except Exception as e:
            raise RuntimeError(f"Pipeline can't transform texts: {e}")
        return obj
    if FEATS_PATH.exists() and CLF_PATH.exists():
        feats = joblib.load(FEATS_PATH)
        clf = joblib.load(CLF_PATH)
        pipe = SKPipeline([("feats", feats), ("clf", clf)])
        try:
            _ = pipe.decision_function(["hello world"])
        except Exception as e:
            raise RuntimeError(f"Reconstructed pipeline can't transform texts: {e}")
        return pipe
    if CLF_PATH.exists():
        raise RuntimeError("Found only classifier. Provide a fitted text transformer or save a full pipeline.")
    raise RuntimeError("No usable model files found in models/.")

def predict_margin(texts, pipe):
    m = pipe.decision_function(texts)
    if isinstance(m, list):
        m = np.asarray(m)
    classes = pipe.named_steps["clf"].classes_
    base_preds = pipe.predict(texts)
    if m.ndim == 1:
        margins = m
        pred_if_pos_cls1 = np.where(margins >= 0, classes[1], classes[0])
        pred_if_pos_cls0 = np.where(margins >= 0, classes[0], classes[1])
        agree1 = np.sum(pred_if_pos_cls1 == base_preds)
        agree0 = np.sum(pred_if_pos_cls0 == base_preds)
        if agree1 >= agree0:
            pos_label, neg_label = classes[1], classes[0]
        else:
            pos_label, neg_label = classes[0], classes[1]
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

st.title("🛒 Fake Product Review Detector")
st.write("Paste a review and I’ll predict whether it’s a **Fake review** or an **Original review** using TF-IDF + Linear SVM.")

try:
    pipe = load_pipe()
except Exception as e:
    st.error(f"Could not load a usable model from {MODELS_DIR}.\n\n{e}")
    st.stop()

classes = pipe.named_steps["clf"].classes_
if len(classes) != 2:
    st.warning(f"Expected binary classes, found: {classes}")

default_th = 0.0
if THR_PATH.exists():
    try:
        default_th = float(json.load(open(THR_PATH))["threshold"])
    except Exception:
        pass

st.sidebar.header("Inference Settings")
th = st.sidebar.slider(
    "Decision threshold (margin)", -2.0, 2.0, 0.8, 0.01,
    help="> 0 favors calling a review Fake.", key="th_slider"
)
show_margin = st.sidebar.checkbox("Show raw margin", value=True)
st.sidebar.markdown("---")
st.sidebar.write("Batch mode available below the text box.")

if "single_pred_text" not in st.session_state:
    st.session_state.single_pred_text = None
if "single_margin" not in st.session_state:
    st.session_state.single_margin = None

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

if st.session_state.single_pred_text is not None:
    st.subheader(f"Prediction: **{st.session_state.single_pred_text}**")
    if show_margin and st.session_state.single_margin is not None:
        st.caption(f"Margin: {st.session_state.single_margin:.4f}  |  Threshold: {th:.2f}")

st.markdown("---")
st.subheader("📦 Batch predictions (CSV)")

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
