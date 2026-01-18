from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
import joblib
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from scheduler import start_scheduler


app = FastAPI(title="Postnatal Depression Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EXPECTED_FEATURES = [
    "Age",
    "Feeling sad or Tearful",
    "Irritable towards baby & partner",
    "Trouble sleeping at night",
    "Problems concentrating or making decision",
    "Overeating or loss of appetite",
    "Feeling anxious",
    "Feeling of guilt",
    "Problems of bonding with baby",
    "Suicide attempt",
]

MAPPING = {
    "yes": 2,
    "no": 0,
    "Two or more days a week": 2.5,
    "sometimes": 1.5,
    "maybe": 1,
    "always": 3
}

DEPRESSION_THRESHOLD = 7

MODEL_PATH = "stacking_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    load_error = str(e)

class SingleRecord(BaseModel):
    Timestamp: Optional[Any] = None
    Age: Optional[Any] = None
    Feeling_sad_or_Tearful: Optional[Any] = None
    Irritable_towards_baby_and_partner: Optional[Any] = None
    Trouble_sleeping_at_night: Optional[Any] = None
    Problems_concentrating_or_making_decision: Optional[Any] = None
    Overeating_or_loss_of_appetite: Optional[Any] = None
    Feeling_anxious: Optional[Any] = None
    Feeling_of_guilt: Optional[Any] = None
    Problems_of_bonding_with_baby: Optional[Any] = None
    Suicide_attempt: Optional[Any] = None

class BatchRequest(BaseModel):
    records: List[dict]

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    rename_map = {
        "feeling_sad_or_tearful": "Feeling sad or Tearful",
        "feeling sad or tearful": "Feeling sad or Tearful",
        "irritable_towards_baby_and_partner": "Irritable towards baby & partner",
        "irritable towards baby & partner": "Irritable towards baby & partner",
        "trouble_sleeping_at_night": "Trouble sleeping at night",
        "trouble sleeping at night": "Trouble sleeping at night",
        "problems_concentrating_or_making_decision": "Problems concentrating or making decision",
        "problems concentrating or making decision": "Problems concentrating or making decision",
        "overeating_or_loss_of_appetite": "Overeating or loss of appetite",
        "overeating or loss of appetite": "Overeating or loss of appetite",
        "feeling_anxious": "Feeling anxious",
        "feeling_of_guilt": "Feeling of guilt",
        "problems_of_bonding_with_baby": "Problems of bonding with baby",
        "suicide_attempt": "Suicide attempt",
        "age": "Age"
    }

    current_cols = list(df.columns)
    for col in current_cols:
        key = col.strip().lower()
        if key in rename_map:
            df = df.rename(columns={col: rename_map[key]})

    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    for col in df.columns:
        if df[col].dtype == object or df[col].dtype == "O" or df[col].dtype == "string":
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].map(MAPPING).fillna(df[col])
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    lower_exclude = {"age", "depression_label", "depression_score"}
    symptom_cols = [c for c in df.columns if c.strip().lower() not in lower_exclude]
    df["depression_score"] = df[symptom_cols].sum(axis=1)

    X = df.copy()
    if "depression_label" in X.columns:
        X = X.drop(columns=["depression_label"])
    if "depression_score" in X.columns:
        X = X.drop(columns=["depression_score"])

    X = X[EXPECTED_FEATURES].astype(float)

    return X, df["depression_score"]
@app.get("/health")
def health():
    return {"status": "alive"}
@app.on_event("startup")
def on_startup():
    global scheduler
    scheduler = start_scheduler()

@app.on_event("shutdown")
def on_shutdown():
    if scheduler:
        scheduler.shutdown()

@app.get("/")
def root():
    return {"message": "Postnatal depression prediction API. POST /predict or /predict_batch"}

@app.post("/predict")
def predict_single(payload: dict):
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")

    try:
        df = pd.DataFrame([payload])
        X, depression_scores = preprocess_dataframe(df)
        preds = model.predict(X)
        result = {"predicted_label": int(preds[0])}

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            result["probability_class_1"] = float(proba[0][:].tolist()[-1]) if proba.shape[1] > 1 else float(proba[0][0])

        result["depression_score"] = float(depression_scores.iloc[0])
        result["rule_label_threshold_7"] = int(depression_scores.iloc[0] >= DEPRESSION_THRESHOLD)

        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")

@app.post("/predict_batch")
def predict_batch(req: BatchRequest):
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")

    try:
        df = pd.DataFrame(req.records)
        X, depression_scores = preprocess_dataframe(df)
        preds = model.predict(X)
        out = []
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)

        for i in range(len(X)):
            item = {
                "index": int(i),
                "predicted_label": int(preds[i]),
                "depression_score": float(depression_scores.iloc[i]),
                "rule_label_threshold_7": int(depression_scores.iloc[i] >= DEPRESSION_THRESHOLD)
            }
            if proba is not None:
                item["probability_class_1"] = float(proba[i][:].tolist()[-1]) if proba.shape[1] > 1 else float(proba[i][0])
            out.append(item)
        return {"predictions": out}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")
