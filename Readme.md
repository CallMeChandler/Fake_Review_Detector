Fake Review Detector 🕵️‍♂️🛒

Detect Fake vs Original product reviews with a lightweight NLP pipeline (no deep learning).
Built with scikit-learn and Streamlit; ships a single saved pipeline you can load and run instantly.

✨ Highlights

Model: Linear SVM on a multi-view text representation

Word view: 1–2 gram presence (binary) – keeps common-but-genuine words useful

Character view: char_wb 4–5 grams (light) – captures style/elongation/punctuation

Fused with FeatureUnion and gentle weights (content dominates, style nudges)

False-positive control: Class weights favor Original on the margin boundary

Practical decision control: Margin threshold slider to tune precision/recall tradeoff live

Batch mode: Upload CSV → get predictions + margins back

📂 Repo structure
.
├─ app.py                      # Streamlit app (single & batch inference UI)
├─ models/
│  ├─ text_svm_pipeline_new.joblib   # Saved sklearn pipeline (vectorizers + LinearSVC)
├─ data/                       # (optional) keep your CSV/datasets here
│  └─ Reviews_Dataset.csv
├─ assets/
│  └─ screenshots/             # put your UI screenshots here (see below)
└─ Fake_Review_Detection.ipynb    # training notebook
└─ README.md

![Home](assets/screenshots/1_home.png)
![Single Prediction](assets/screenshots/2_single.png)
![Batch Upload](assets/screenshots/3_batch.png)
![Batch Results](assets/screenshots/4_results.png)

🚀 Quickstart

Create a venv (recommended):

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate


Install deps (pin sklearn to the same version used at train time to avoid pickle issues):

pip install streamlit scikit-learn joblib pandas numpy


Place model files:

Put text_svm_pipeline_new.joblib inside models/.

Optional: create models/threshold.json with {"threshold": 0.12} (or your tuned value).

Run the app:

streamlit run app.py


Open the local URL → paste a review → Predict.

🧠 How it works (Model)

Task: Binary classification
Labels: OR = Fake review, CG = Original review (mapped to user-friendly labels in the UI)

Representation (FeatureUnion):

Word (content): CountVectorizer(ngram_range=(1,2), binary=True, min_df=3)
Presence features keep common positive words (e.g., “good”) informative without IDF down-weighting.

Character (style): TfidfVectorizer(analyzer='char_wb', ngram_range=(4,5), use_idf=False, sublinear_tf=True, min_df=6)
Adds shallow stylistic clues (elongation, punctuation bursts, caps) within word boundaries.
Given lower weight so it never overwhelms content.

Classifier: LinearSVC(C=1.0, class_weight={'CG':1.3, 'OR':1.0})
Slightly penalizes misclassifying Original as Fake (reduces harmful false positives).

Decision: We use the SVM margin (decision_function) and apply a threshold.
Default cutoff can be tuned on a validation split and stored in models/threshold.json.

Training/Eval: See notebooks/Fake_Review_Detection.ipynb for the full pipeline, splitting, and metrics.
(Accuracy/precision/recall are reported on a held-out test split.)

🖥️ How it works (App)

Loads a fitted pipeline (models/text_svm_pipeline_new.joblib) and optionally a tuned threshold (models/threshold.json).

Single-review form: transforms text → gets margin → compares to slider threshold → shows Fake review/Original review + margin.

Batch mode: upload CSV, pick a text column, and download predictions with margins.
This behavior (loading, decision via decision_function, UI, and batch flow) is implemented in app.py. 

⚙️ Configuration & Tips

Threshold: Use the sidebar to raise/lower the margin cutoff.

Higher threshold ⇒ stricter “Fake” (fewer false positives on genuine reviews, but may miss some fakes).

Lower threshold ⇒ more sensitive to “Fake” (higher recall, lower precision).

Versioning: Ensure the same scikit-learn version is used to train and serve (pickles aren’t always forward/back compatible).

Small texts: One-word reviews (“good”) are low-information; if needed, raise threshold slightly.
(You can also implement a length-aware threshold in the app.)

Dataset: Trained on an e-commerce reviews dataset (e.g., Amazon/Kaggle). Swap in any CSV; retrain the notebook to update the pipeline.

📊 Reproducible Training (Notebook)

Split: Stratified train/test to preserve class ratios

Features: Word 1–2 gram (binary) + char_wb 4–5 gram (light)

Classifier: LinearSVC with class weights

Tuning: Pick a margin threshold on a validation fold to hit your target (e.g., Fake precision ≥ 90%)

Export: Save the fitted pipeline as models/text_svm_pipeline_new.joblib and (optionally) models/threshold.json

🧪 Limitations

Margin is not a probability; for calibrated probabilities use CalibratedClassifierCV or Logistic Regression.

Extremely short inputs can be ambiguous by nature; thresholding helps.

If you change tokenization/feature settings, retrain and re-export the pipeline.

📄 License

MIT (or your choice). Add a short data usage note if your dataset has its own license.

🙏 Acknowledgments

scikit-learn maintainers

Streamlit team

Public Dataset:
https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset?resource=download
