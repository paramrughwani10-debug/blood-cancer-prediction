"""
model_utils.py
Rebuilds the Orange workflow (BLOOD_CANCER_.ows) in Python with scikit-learn.

Orange widget                 ->  Code here
-----------------------------------------------------------------
File (TRAIN / TEST)           ->  load_data()
Select Columns                ->  FEATURES, TARGET
Preprocess (scale)            ->  ColumnTransformer + MinMaxScaler
Random Forest                 ->  RandomForestClassifier
Gradient Boosting             ->  GradientBoostingClassifier
kNN                           ->  KNeighborsClassifier
Test and Score                ->  evaluate()
Predictions                   ->  predict()
"""
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TARGET = "Diagnosis"
FEATURES = ["Age", "Gender", "BMI", "Smoking", "GeneticRisk",
            "PhysicalActivity", "AlcoholIntake", "CancerHistory"]
# Columns that Orange's Preprocess widget scales (the 0/1 yes-no columns are left as they are)
SCALE_COLS = ["Age", "BMI", "GeneticRisk", "PhysicalActivity", "AlcoholIntake"]


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "Cancer_Train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "Cancer_Test.csv"))
    live = pd.read_csv(os.path.join(DATA_DIR, "Cancer_Unseen_LiveData.csv"))
    return train, test, live


def _make_pipeline(model):
    prep = ColumnTransformer([("scale", MinMaxScaler(), SCALE_COLS)],
                             remainder="passthrough")
    return Pipeline([("prep", prep), ("model", model)])


def get_models():
    """Same settings as the Orange widgets."""
    return {
        "Random Forest": _make_pipeline(RandomForestClassifier(
            n_estimators=10, min_samples_split=5, random_state=0)),
        "Gradient Boosting": _make_pipeline(GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)),
        "kNN": _make_pipeline(KNeighborsClassifier(
            n_neighbors=5, metric="euclidean", weights="uniform")),
    }


def train_and_evaluate(train, test):
    """Train on TRAIN file, score on TEST file (like Test and Score)."""
    models = get_models()
    rows, matrices = [], {}
    for name, pipe in models.items():
        pipe.fit(train[FEATURES], train[TARGET])
        pred = pipe.predict(test[FEATURES])
        proba = pipe.predict_proba(test[FEATURES])[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(test[TARGET], pred),
            "Precision": precision_score(test[TARGET], pred, zero_division=0),
            "Recall": recall_score(test[TARGET], pred, zero_division=0),
            "F1": f1_score(test[TARGET], pred, zero_division=0),
            "AUC": roc_auc_score(test[TARGET], proba),
        })
        matrices[name] = confusion_matrix(test[TARGET], pred, labels=[0, 1])
    scores = pd.DataFrame(rows).set_index("Model")
    return models, scores, matrices


def predict(pipe, df):
    """Return a copy of df with predicted Diagnosis and probability."""
    out = df.copy()
    out["Predicted_Diagnosis"] = pipe.predict(df[FEATURES])
    out["Probability_of_1"] = pipe.predict_proba(df[FEATURES])[:, 1].round(3)
    return out


def check_columns(df):
    missing = [c for c in FEATURES if c not in df.columns]
    return missing
