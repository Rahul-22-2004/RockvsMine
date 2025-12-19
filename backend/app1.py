import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import List, Optional, Any
import uvicorn
import os
from datetime import datetime
import warnings
import joblib
import base64
from io import BytesIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ================= CONFIG =================
DATASET_URL = "https://drive.google.com/uc?export=download&id=1s_c44vosOb_Xwejx9128_8yL8Wr6-3WV"
MODEL_ARTIFACT_PATH = "./models"
OOD_SIMILARITY_THRESHOLD = 90.0
os.makedirs(MODEL_ARTIFACT_PATH, exist_ok=True)

# ================= GLOBALS =================
best_model = None
scaler = None
pca = None
training_data_stats = {}
model_name = ""
prediction_history = []
roc_curve_base64 = None

# ================= FASTAPI =================
app = FastAPI(
    title="Rock vs Mine ML API",
    description="Logistic Regression + PCA with ROC Curve",
    version="3.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= SCHEMAS =================
class InputData(BaseModel):
    values: List[Any]

    @validator("values")
    def validate_values(cls, v):
        if len(v) != 60:
            raise ValueError("Expected exactly 60 values")
        return [float(x) for x in v]

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    original_confidence: float
    adjusted_confidence: float
    penalty_or_boost: Optional[float]
    in_distribution: bool
    similarity_score: float
    reason: str
    model_used: str
    timestamp: str
    warning: Optional[str]

# ================= UTILITIES =================
def fetch_dataset(url):
    df = pd.read_csv(url, header=None)
    print("=" * 80)
    print(" DATASET LOADED")
    print(f" Shape of dataset: {df.shape}")  # ✅ REQUIRED BY YOU
    print("=" * 80)
    return df

def calculate_training_statistics(X):
    stats = {"per_feature": {}}
    for i in range(X.shape[1]):
        stats["per_feature"][i] = {
            "min": float(X[:, i].min()),
            "max": float(X[:, i].max())
        }
    return stats

def calculate_similarity(input_scaled):
    in_range = 0
    for i, v in enumerate(input_scaled):
        if training_data_stats["per_feature"][i]["min"] <= v <= training_data_stats["per_feature"][i]["max"]:
            in_range += 1
    similarity = (in_range / len(input_scaled)) * 100
    return similarity, similarity >= OOD_SIMILARITY_THRESHOLD

def adjust_confidence(original, similarity):
    if similarity >= 95:
        boost = (100 - original) * 0.4
        return min(95, original + boost), boost, "High similarity"
    elif similarity >= 70:
        penalty = (100 - similarity) * 0.3
        return max(50, original - penalty), penalty, "Moderate similarity"
    else:
        penalty = (100 - similarity) * 0.8
        return max(45, original - penalty), penalty, "Low similarity"

def generate_roc_curve(y_true, y_prob, model_name):
    global roc_curve_base64
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    buf.seek(0)
    roc_curve_base64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)

# ================= TRAIN MODEL =================
def train_model():
    global best_model, scaler, pca, training_data_stats, model_name

    df = fetch_dataset(DATASET_URL)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    y_bin = (y == "M").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ✅ PCA FOR BETTER AUC
    pca = PCA(n_components=20, random_state=42)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    training_data_stats = calculate_training_statistics(X_train)

    models = {
        "LogReg_ElasticNet": LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            l1_ratio=0.5,
            C=0.3,
            max_iter=6000
        ),
        "LogReg_Balanced": LogisticRegression(
            solver="lbfgs",
            C=0.2,
            class_weight="balanced",
            max_iter=5000
        )
    }

    best_auc = -1

    for name, model in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_bin[y_test.index if hasattr(y_test,'index') else range(len(y_test))], probs)

        if auc_score > best_auc:
            best_auc = auc_score
            best_model = model
            model_name = name
            best_probs = probs
            best_y = (y_test == "M").astype(int)

    generate_roc_curve(best_y, best_probs, model_name)

    joblib.dump(best_model, f"{MODEL_ARTIFACT_PATH}/best_model.joblib")
    joblib.dump(scaler, f"{MODEL_ARTIFACT_PATH}/scaler.joblib")
    joblib.dump(pca, f"{MODEL_ARTIFACT_PATH}/pca.joblib")

    print(f" Best Model: {model_name} | ROC-AUC: {best_auc:.4f}")

# ================= STARTUP =================
@app.on_event("startup")
async def startup():
    train_model()

# ================= ERROR HANDLER =================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# ================= ENDPOINTS =================
@app.get("/")
def root():
    return {
        "status": "online",
        "model_loaded": best_model is not None,
        "roc_curve_available": roc_curve_base64 is not None
    }

@app.get("/roc-curve")
def roc_curve_endpoint():
    return {
        "roc_curve_image": f"data:image/png;base64,{roc_curve_base64}",
        "model_name": model_name,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData):
    X = scaler.transform(np.array(data.values).reshape(1, -1))
    X = pca.transform(X)

    prob = best_model.predict_proba(X)[0]
    pred = np.argmax(prob)

    original_conf = prob[pred] * 100
    similarity, in_dist = calculate_similarity(X.flatten())
    adjusted, delta, reason = adjust_confidence(original_conf, similarity)

    label = "Mine" if pred == 1 else "Rock"

    response = {
        "prediction": label,
        "confidence": round(adjusted, 2),
        "original_confidence": round(original_conf, 2),
        "adjusted_confidence": round(adjusted, 2),
        "penalty_or_boost": round(delta, 2),
        "in_distribution": in_dist,
        "similarity_score": round(similarity, 2),
        "reason": reason,
        "model_used": model_name,
        "timestamp": datetime.now().isoformat(),
        "warning": None if in_dist else "Input outside distribution"
    }

    prediction_history.append(response)
    return response

# ================= RUN =================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
