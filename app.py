"""
Blood Cancer Risk Prediction - Class 12 AI Capstone
Streamlit website for the Orange workflow (BLOOD_CANCER_.ows).
Run with:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

import model_utils as mu

st.set_page_config(page_title="Blood Cancer Prediction", page_icon="🩸", layout="wide")

# ---- Label text for 0/1 columns (edit these to match how YOUR dataset was coded) ----
LABELS = {
    "Gender": {0: "0", 1: "1"},
    "Smoking": {0: "0 - No", 1: "1 - Yes"},
    "CancerHistory": {0: "0 - No", 1: "1 - Yes"},
    "GeneticRisk": {0: "0 - Low", 1: "1 - Medium", 2: "2 - High"},
    "Diagnosis": {0: "0 - Negative", 1: "1 - Positive"},
}

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
h1, h2, h3 {color: #1b4f8a;}
.result-box {padding: 1rem 1.3rem; border-radius: 10px; font-size: 1.15rem;
             background: #eaf2fb; border-left: 6px solid #1b4f8a;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    return mu.load_data()


@st.cache_resource
def get_trained():
    """Train once, reuse on every click (like running the Orange workflow once)."""
    train, test, _ = get_data()
    return mu.train_and_evaluate(train, test)


train_df, test_df, live_df = get_data()
models, scores, matrices = get_trained()
best_model = scores["F1"].idxmax()

page = st.sidebar.radio("Go to", [
    "Home", "Dataset Explorer", "Model Comparison",
    "Predict One Patient", "Batch Prediction", "About & Disclaimer"])
st.sidebar.info("Educational student project. Not a medical diagnosis tool.")

# ------------------------------------------------------------------ HOME
if page == "Home":
    st.title("🩸 Blood Cancer Prediction System")
    st.write("A machine learning project that predicts the **Diagnosis** (0 or 1) "
             "from patient lifestyle and health details. The models were built "
             "in the Orange Data Mining tool and rebuilt here as a website.")
    st.subheader("Workflow")
    cols = st.columns(5)
    steps = [("1. Data", "Train, Test and new patient files"),
             ("2. Select", "8 features + Diagnosis as target"),
             ("3. Preprocess", "Scale the numeric columns"),
             ("4. Train", "Random Forest, Gradient Boosting, kNN"),
             ("5. Predict", "Test scores and new patient predictions")]
    for c, (t, d) in zip(cols, steps):
        c.metric(t, "")
        c.caption(d)
    a, b, c3 = st.columns(3)
    a.metric("Training rows", len(train_df))
    b.metric("Testing rows", len(test_df))
    c3.metric("Best model (by F1)", best_model)

# ------------------------------------------------------- DATASET EXPLORER
elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    which = st.radio("Dataset", ["Training", "Testing"], horizontal=True)
    df = train_df if which == "Training" else test_df
    st.dataframe(df, use_container_width=True, height=300)
    st.subheader("Diagnosis distribution")
    counts = df[mu.TARGET].map(LABELS["Diagnosis"]).value_counts()
    st.bar_chart(counts)
    st.subheader("Feature charts")
    feat = st.selectbox("Choose a feature", mu.FEATURES)
    if feat in mu.SCALE_COLS and df[feat].nunique() > 3:
        binned = pd.cut(df[feat], bins=10).value_counts().sort_index()
        binned.index = binned.index.astype(str)
        st.bar_chart(binned)
    else:
        st.bar_chart(df[feat].value_counts().sort_index())
    st.caption("Average of each feature for each Diagnosis group:")
    st.dataframe(df.groupby(mu.TARGET)[mu.FEATURES].mean().round(2))

# ------------------------------------------------------- MODEL COMPARISON
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.write("Models trained on the Training file and scored on the Testing file "
             "(same as the *Test and Score* widget).")
    st.dataframe(scores.style.format("{:.3f}")
                 .highlight_max(axis=0, color="#cfe8cf"), use_container_width=True)
    st.success(f"Best model by F1 score: **{best_model}**")
    st.bar_chart(scores[["Accuracy", "F1", "AUC"]])
    st.subheader("Confusion matrices")
    cols = st.columns(3)
    for col, (name, cm) in zip(cols, matrices.items()):
        col.markdown(f"**{name}**")
        col.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"],
                                   columns=["Predicted 0", "Predicted 1"]))

# ------------------------------------------------------ PREDICT ONE PATIENT
elif page == "Predict One Patient":
    st.title("Predict for One Patient")
    model_name = st.selectbox("Model", list(models), index=list(models).index(best_model))
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 18, 100, 45)
        bmi = st.slider("BMI", 10.0, 50.0, 25.0, 0.1)
        activity = st.slider("Physical activity (hours/week)", 0.0, 12.0, 4.0, 0.1)
        alcohol = st.slider("Alcohol intake (units/week)", 0.0, 6.0, 2.0, 0.1)
    with c2:
        gender = st.selectbox("Gender", [0, 1], format_func=LABELS["Gender"].get)
        smoking = st.selectbox("Smoking", [0, 1], format_func=LABELS["Smoking"].get)
        genetic = st.selectbox("Genetic risk", [0, 1, 2], format_func=LABELS["GeneticRisk"].get)
        history = st.selectbox("Cancer history", [0, 1], format_func=LABELS["CancerHistory"].get)
    if st.button("Predict", type="primary"):
        row = pd.DataFrame([{"Age": age, "Gender": gender, "BMI": bmi, "Smoking": smoking,
                             "GeneticRisk": genetic, "PhysicalActivity": activity,
                             "AlcoholIntake": alcohol, "CancerHistory": history}])
        res = mu.predict(models[model_name], row).iloc[0]
        p = float(res["Probability_of_1"])
        st.markdown(f"<div class='result-box'>Predicted Diagnosis: "
                    f"<b>{LABELS['Diagnosis'][int(res['Predicted_Diagnosis'])]}</b></div>",
                    unsafe_allow_html=True)
        st.write("Confidence that Diagnosis = 1:")
        st.progress(p, text=f"{p:.0%}")
        st.caption("This is a student project prediction, not medical advice.")

# ---------------------------------------------------------- BATCH PREDICTION
elif page == "Batch Prediction":
    st.title("Batch Prediction")
    st.write("Upload a CSV with the columns: " + ", ".join(mu.FEATURES))
    model_name = st.selectbox("Model", list(models), index=list(models).index(best_model))
    up = st.file_uploader("Upload CSV", type="csv")
    use_demo = st.checkbox("Use the sample file (Cancer_Unseen_LiveData.csv)", value=up is None)
    data = None
    if up is not None:
        try:
            data = pd.read_csv(up)
        except Exception:
            st.error("Could not read that file. Please upload a valid CSV.")
    elif use_demo:
        data = live_df
    if data is not None:
        missing = mu.check_columns(data)
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        elif data[mu.FEATURES].isna().any().any():
            st.error("The file has empty cells. Please fill them and upload again.")
        else:
            try:
                out = mu.predict(models[model_name], data)
                st.dataframe(out, use_container_width=True)
                st.download_button("Download results (CSV)", out.to_csv(index=False),
                                   "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Prediction failed - check that all values are numbers. ({e})")

# ------------------------------------------------------------------ ABOUT
else:
    st.title("About & Disclaimer")
    st.write("This website is the user interface for a Class 12 AI capstone project. "
             "The machine learning workflow was designed in **Orange Data Mining** "
             "and re-created with **Python (scikit-learn)** and **Streamlit**.")
    st.markdown("""
| Orange widget | In this website |
|---|---|
| File (Train / Test / New data) | `load_data()` and the Batch upload |
| Select Columns | `FEATURES` and `TARGET` in `model_utils.py` |
| Preprocess | MinMaxScaler on numeric columns |
| Random Forest, Gradient Boosting, kNN | Same models, same settings |
| Test and Score | Model Comparison page |
| Predictions | Predict One Patient and Batch Prediction pages |
""")
    st.warning("**Disclaimer:** This is an educational project made by students. "
               "It is NOT a medical diagnosis tool and must not be used to make "
               "health decisions. Please consult a doctor for any health concern.")
