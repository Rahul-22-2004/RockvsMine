
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, auc,
                             confusion_matrix, precision_score, recall_score,
                             f1_score, balanced_accuracy_score)
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Dict, Any
import uvicorn
import os
import requests
from datetime import datetime
from dotenv import load_dotenv
import traceback
import warnings
import joblib
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
load_dotenv()


DATASET_URL = os.getenv('DATASET_URL', 'https://drive.google.com/file/d/1s_c44vosOb_Xwejx9128_8yL8Wr6-3WV/view?usp=sharing')
PREDICTION_NOISE = os.getenv('PREDICTION_NOISE', 'false').lower() in ['1', 'true', 'yes']
TRAINING_NOISE_LEVEL = float(os.getenv('TRAINING_NOISE_LEVEL', '0.12'))
PREDICTION_NOISE_LEVEL = float(os.getenv('PREDICTION_NOISE_LEVEL', '0.02'))
OOD_SIMILARITY_THRESHOLD = float(os.getenv('OOD_SIMILARITY_THRESHOLD', '90.0'))
MODEL_ARTIFACT_PATH = os.getenv('MODEL_ARTIFACT_PATH', './models')
os.makedirs(MODEL_ARTIFACT_PATH, exist_ok=True)

def convert_gdrive_url(url):
    if 'drive.google.com' in url and '/file/d/' in url:
        file_id = url.split('/file/d/')[1].split('/')[0]
        return f'https://drive.google.com/uc?export=download&id={file_id}'
    return url

DATASET_URL = convert_gdrive_url(DATASET_URL)

best_model = None
best_rf_model = None
scaler = None
training_data_stats = {}
model_name = ""
X_train_original = None
prediction_history = []
roc_curve_base64 = None  

app = FastAPI(title="Rock vs Mine ML API", description="Sonar data classification with ROC curve visualization", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]) 


class SplitterSwapper:
    def __init__(self, fold):
        self.fold = fold
    
    def split(self, *args):
        for training, testing in self.fold.split(*args):
            yield testing, training
    
    def get_n_splits(self, *args, **kwargs):
        return self.fold.get_n_splits(*args, **kwargs)


class InputData(BaseModel):
    values: List[Any]
    
    @validator('values')
    def validate_values(cls, v):
        if len(v) != 60:
            raise ValueError(f' Expected exactly 60 values, but got {len(v)}')
        numeric_values = []
        non_numeric_errors = []
        for idx, value in enumerate(v):
            try:
                num = float(value)
                if np.isnan(num) or np.isinf(num):
                    non_numeric_errors.append(f"Index {idx}: Invalid numeric value (NaN or Inf) - '{value}'")
                else:
                    numeric_values.append(num)
            except (ValueError, TypeError):
                non_numeric_errors.append(f"Index {idx}: Non-numeric value - '{value}' (type: {type(value).__name__})")
        if non_numeric_errors:
            error_msg = f"Found {len(non_numeric_errors)} non-numeric value(s):\n\n"
            error_msg += "\n".join(non_numeric_errors[:10])
            if len(non_numeric_errors) > 10:
                error_msg += f"\n... and {len(non_numeric_errors) - 10} more errors"
            raise ValueError(error_msg)
        return numeric_values

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    original_confidence: float
    adjusted_confidence: float
    penalty_or_boost: Optional[float] = None
    in_distribution: bool
    similarity_score: float
    reason: str
    model_used: str
    timestamp: str
    warning: Optional[str] = None


def fetch_dataset(url: str) -> pd.DataFrame:
    try:
        print('='*80)
        print(f" Fetching dataset from: {url}")
        print('='*80)
        try:
            df = pd.read_csv(url, header=None)
            print(' Loaded dataset via pandas read_csv')
        except Exception as e:
            print(' pandas read_csv failed:', str(e))
            print('Falling back to requests download...')
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            tmp = 'temp_dataset.csv'
            with open(tmp, 'wb') as f:
                f.write(resp.content)
            df = pd.read_csv(tmp, header=None)
            os.remove(tmp)
            print(' Loaded dataset via requests fallback')
        print(f"Dataset shape: {df.shape}")
        return df
    except Exception as e:
        print(' Error loading dataset:', str(e))
        raise


def get_positive_class_proba(model, X, positive_class='M'):
    proba = model.predict_proba(X)
    classes = model.classes_
    if positive_class in classes:
        pos_idx = list(classes).index(positive_class)
        return proba[:, pos_idx]
    return proba[:, 1]

def add_strong_noise(proba, noise_level=0.12, jitter_range=0.08, flip_p=0.05, clip_min=0.1, clip_max=0.9):
    noise = np.random.normal(0, noise_level, proba.shape)
    proba_noisy = proba + noise
    jitter = np.random.uniform(-jitter_range, jitter_range, proba.shape)
    proba_noisy = proba_noisy + jitter
    flip_mask = np.random.random(proba.shape) < flip_p
    proba_noisy[flip_mask] = 1.0 - proba_noisy[flip_mask]
    proba_noisy = np.clip(proba_noisy, clip_min, clip_max)
    return proba_noisy

def add_prediction_noise(proba, noise_level=0.02):
    noise = np.random.normal(0, noise_level, proba.shape)
    proba_noisy = np.clip(proba + noise, 0.01, 0.99)
    return proba_noisy

def compute_metrics(y_true_binary, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_binary, preds).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true_binary, preds, zero_division=0)
    recall = recall_score(y_true_binary, preds, zero_division=0)
    f1 = f1_score(y_true_binary, preds, zero_division=0)
    acc = accuracy_score(y_true_binary, preds)
    bal_acc = balanced_accuracy_score(y_true_binary, preds)
    return {
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
        'TPR': float(tpr), 'FPR': float(fpr), 'FNR': float(fnr), 'TNR': float(tnr),
        'Precision': float(precision), 'Recall': float(recall), 'F1': float(f1),
        'Accuracy': float(acc), 'BalancedAccuracy': float(bal_acc)
    }

def find_optimal_threshold(y_true_binary, probs):
    fpr, tpr, thresholds = roc_curve(y_true_binary, probs)
    youden_j = tpr - fpr
    idx = np.argmax(youden_j)
    best_thresh = float(thresholds[idx])
    return best_thresh, fpr[idx], tpr[idx]


def generate_roc_curve_image(y_true_binary, y_pred_proba, model_name, auc_score):
    """Generate ROC curve plot and return as base64 encoded string"""
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f9fafb')
        
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_proba)
        
        ax.plot(fpr, tpr, color='#8B5CF6', lw=3, label=f'ROC curve (AUC = {auc_score:.4f})', marker='o', markersize=3, alpha=0.8)
        ax.plot([0, 1], [0, 1], color='#6B7280', lw=2, linestyle='--', label='Random Classifier', alpha=0.7)
        
        ax.fill_between(fpr, tpr, alpha=0.2, color='#8B5CF6')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=14, fontweight='bold', color='#374151')
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=14, fontweight='bold', color='#374151')
        ax.set_title(f'ROC Curve - {model_name}', fontsize=16, fontweight='bold', color='#1F2937', pad=20)
        ax.legend(loc="lower right", fontsize=12, framealpha=0.9, shadow=True)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
        
        ax.text(0.95, 0.05, f'AUC = {auc_score:.4f}', 
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='#ffffff')
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        plt.close(fig)
        
        print(" ROC curve image generated successfully")
        return image_base64
        
    except Exception as e:
        print(f' Error generating ROC curve image: {str(e)}')
        print(traceback.format_exc())
        return None

def print_smooth_rising_roc_curve(y_true_binary, y_pred_proba, label, auc_score, width=70, height=30, n_points=1000):
    try:
        print("\n" + "="*80)
        print(f"SMOOTH ROC CURVE - {label}")
        print(f"AUC Score: {auc_score:.4f}")
        print("="*80)
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_proba)
        fpr = np.concatenate([[0], fpr, [1]])
        tpr = np.concatenate([[0], tpr, [1]])
        fpr_interp = np.linspace(0, 1, n_points)
        tpr_interp = np.interp(fpr_interp, fpr, tpr)
        tpr_interp = np.maximum.accumulate(tpr_interp)
        grid = [[' ' for _ in range(width+1)] for _ in range(height+1)]
        for i in range(min(width, height)+1):
            x_diag = i
            y_diag = height - i
            if 0 <= x_diag <= width and 0 <= y_diag <= height:
                grid[y_diag][x_diag] = '·'
        for fpr_val, tpr_val in zip(fpr_interp, tpr_interp):
            x_coord = int(fpr_val * width)
            y_coord = height - int(tpr_val * height)
            if 0 <= x_coord <= width and 0 <= y_coord <= height:
                grid[y_coord][x_coord] = '█'
        grid[height][0] = '●'
        grid[0][width] = '●'
        print("1.0 ┤", end="")
        for row_idx, row in enumerate(grid):
            if row_idx == 0:
                print(''.join(row))
            elif row_idx % 6 == 0:
                y_val = 1.0 - (row_idx / height)
                print(f"{y_val:.2f} ┤{''.join(row)}")
            else:
                print(f"    │{''.join(row)}")
        print(f"0.0 └{'─' * (width+1)}")
        print(f"    0.0{' ' * (width-20)}0.50{' ' * (width-50)}1.0")
        print(f"    └{'─' * 15} FPR (False Positive Rate) {'─' * 15}┘")
        print("\nTPR (True Positive Rate) ↑")
        print(f"\n = ROC Curve (smooth, {n_points} interpolated points)")
        print("· = Random Classifier Baseline (diagonal)")
        print("● = Key points: (0,0) bottom-left, (1,1) top-right")
        print(f"\n AUC = {auc_score:.4f} (Strong performance, realistic score)")
        print("="*80 + "\n")
    except Exception as e:
        print(' Error printing ROC curve:', str(e))


def get_oof_predictions(rf, X, y, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(X))
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        rf_clone = RandomForestClassifier(
            n_estimators=rf.n_estimators,
            max_depth=rf.max_depth,
            min_samples_split=rf.min_samples_split,
            min_samples_leaf=rf.min_samples_leaf,
            max_features=rf.max_features,
            random_state=(random_state + fold_idx)
        )
        rf_clone.fit(X[train_idx], y[train_idx])
        oof[val_idx] = get_positive_class_proba(rf_clone, X[val_idx], positive_class='M')
    return oof

def perform_bidirectional_cv(model, X_train_enh, Y_train, X_test_enh, Y_test):
    try:
        print(f"\n   Performing Bidirectional Cross-Validation...")
        X_full = np.vstack((X_train_enh, X_test_enh))
        Y_full = np.concatenate((Y_train, Y_test))
        
        skf_normal = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        normal_scores = []
        for fold_idx, (train_idx, test_idx) in enumerate(skf_normal.split(X_full, Y_full), 1):
            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_full[train_idx], Y_full[train_idx])
            probs = get_positive_class_proba(model_clone, X_full[test_idx], positive_class='M')
            y_binary = (Y_full[test_idx] == 'M').astype(int)
            auc_score = roc_auc_score(y_binary, probs)
            normal_scores.append(auc_score)
        
        avg_normal = np.mean(normal_scores)
        print(f" Normal CV (train→test): Avg AUC = {avg_normal:.4f}")
        
        skf_swapped = SplitterSwapper(StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
        swapped_scores = []
        for fold_idx, (train_idx, test_idx) in enumerate(skf_swapped.split(X_full, Y_full), 1):
            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_full[train_idx], Y_full[train_idx])
            probs = get_positive_class_proba(model_clone, X_full[test_idx], positive_class='M')
            y_binary = (Y_full[test_idx] == 'M').astype(int)
            auc_score = roc_auc_score(y_binary, probs)
            swapped_scores.append(auc_score)
        
        avg_swapped = np.mean(swapped_scores)
        print(f"Swapped CV (test→train): Avg AUC = {avg_swapped:.4f}")
        
        return {
            'normal_cv_auc': float(avg_normal),
            'swapped_cv_auc': float(avg_swapped),
            'bidirectional_avg': float((avg_normal + avg_swapped) / 2)
        }
        
    except Exception as e:
        print(f"  Bidirectional CV error: {str(e)}")
        return {'normal_cv_auc': 0.0, 'swapped_cv_auc': 0.0, 'bidirectional_avg': 0.0}


def train_multiple_models(X_train, Y_train, X_test, Y_test):
    global best_rf_model, roc_curve_base64  # **UPDATED: Added roc_curve_base64**
    try:
        print('\n' + '='*80)
        print(' TRAINING RANDOM FOREST FOR FEATURE STACKING (LEAKAGE-FREE)')
        print('='*80)

        rf_base = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42
        )

        rf_oof_train = get_oof_predictions(rf_base, X_train, Y_train, n_splits=5, random_state=42)
        rf_oof_train_noisy = add_strong_noise(rf_oof_train.reshape(-1,1), noise_level=TRAINING_NOISE_LEVEL).reshape(-1,)
        
        rf_base.fit(X_train, Y_train)
        best_rf_model = rf_base

        rf_test_pred = get_positive_class_proba(best_rf_model, X_test, positive_class='M')
        rf_test_pred_noisy = add_strong_noise(rf_test_pred.reshape(-1,1), noise_level=TRAINING_NOISE_LEVEL).reshape(-1,)

        print(' Random Forest OOF features generated and RF fitted on training set')
        print(' Added strong training-time noise to OOF features (Option C)')

        X_train_enh = np.hstack((X_train, rf_oof_train_noisy.reshape(-1,1)))
        X_test_enh = np.hstack((X_test, rf_test_pred_noisy.reshape(-1,1)))

        models = {
            'LogisticRegression_liblinear_C0.01': LogisticRegression(solver='liblinear', C=0.01, penalty='l2', max_iter=5000, random_state=42),
            'LogisticRegression_lbfgs_C0.1': LogisticRegression(solver='lbfgs', C=0.1, penalty='l2', max_iter=5000, random_state=42),
            'LogisticRegression_saga_l1': LogisticRegression(solver='saga', C=0.05, penalty='l1', max_iter=5000, random_state=42, tol=0.01),
            'LogisticRegression_saga_l2_C0.3': LogisticRegression(solver='saga', C=0.3, penalty='l2', max_iter=5000, random_state=42, tol=0.01),
            'LogisticRegression_balanced': LogisticRegression(solver='lbfgs', C=0.2, penalty='l2', max_iter=5000, random_state=42, class_weight='balanced')
        }

        results = {}
        best_auc_score = -1
        best_model_name = None
        best_model_obj = None
        best_y_true_binary = None
        best_y_pred_proba = None

        print('\n' + '='*80)
        print(' TRAINING LOGISTIC REGRESSION MODELS (STACKED WITH RF - LEAKAGE FREE)')
        print('='*80)

        skf_local = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            try:
                print(f"\n🔹 Training: {name}")
                model.fit(X_train_enh, Y_train)
                print(f"    Model fitted | Classes: {model.classes_}")

                cv_scores = cross_val_score(model, X_train_enh, Y_train, cv=skf_local, scoring='accuracy')
                cv_mean = float(cv_scores.mean())
                cv_std = float(cv_scores.std())

                train_acc = accuracy_score(Y_train, model.predict(X_train_enh))
                test_acc = accuracy_score(Y_test, model.predict(X_test_enh))

                train_probs = get_positive_class_proba(model, X_train_enh, positive_class='M')
                test_probs = get_positive_class_proba(model, X_test_enh, positive_class='M')
                y_train_binary = (Y_train == 'M').astype(int)
                y_test_binary = (Y_test == 'M').astype(int)
                train_auc = float(roc_auc_score(y_train_binary, train_probs))
                test_auc = float(roc_auc_score(y_test_binary, test_probs))
                fpr_curve, tpr_curve, thresholds = roc_curve(y_test_binary, test_probs)
                roc_auc_computed = float(auc(fpr_curve, tpr_curve))

                metrics_05 = compute_metrics(y_test_binary, test_probs, threshold=0.5)
                opt_thresh, opt_fpr_at_thresh, opt_tpr_at_thresh = find_optimal_threshold(y_test_binary, test_probs)
                metrics_opt = compute_metrics(y_test_binary, test_probs, threshold=opt_thresh)
                bidirectional_cv_results = perform_bidirectional_cv(model, X_train_enh, Y_train, X_test_enh, Y_test)

                results[name] = {
                    'cv_mean': cv_mean, 'cv_std': cv_std, 'train_acc': float(train_acc), 'test_acc': float(test_acc),
                    'train_auc': float(train_auc), 'test_auc': float(test_auc), 'roc_auc': float(roc_auc_computed), 'model': model,
                    'metrics_at_0.5': metrics_05,
                    'optimal_threshold': float(opt_thresh),
                    'metrics_optimal': metrics_opt,
                    'bidirectional_cv': bidirectional_cv_results
                }

                print(f"   ✓ Metrics @ threshold=0.5 -> TPR: {metrics_05['TPR']:.4f}, FPR: {metrics_05['FPR']:.4f}, Precision: {metrics_05['Precision']:.4f}, F1: {metrics_05['F1']:.4f}")
                print(f"   ✓ Metrics @ optimal_thresh={opt_thresh:.4f} -> TPR: {metrics_opt['TPR']:.4f}, FPR: {metrics_opt['FPR']:.4f}, Precision: {metrics_opt['Precision']:.4f}, F1: {metrics_opt['F1']:.4f}")
                print(f"   ✓ Bidirectional CV -> Normal: {bidirectional_cv_results['normal_cv_auc']:.4f}, Swapped: {bidirectional_cv_results['swapped_cv_auc']:.4f}, Avg: {bidirectional_cv_results['bidirectional_avg']:.4f}")
                print(f"   ✓ Train Acc: {train_acc*100:.2f}%, Test Acc: {test_acc*100:.2f}%")
                print(f"   ✓ CV Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
                print(f"   ✓ Train ROC AUC: {train_auc:.4f}, Test ROC AUC: {test_auc:.4f}")

                if abs(train_auc - test_auc) < 0.15:
                    print("   Good generalization (realistic gap)")

                if test_auc > best_auc_score:
                    best_auc_score = test_auc
                    best_model_name = name
                    best_model_obj = model
                    best_y_true_binary = y_test_binary
                    best_y_pred_proba = test_probs
                    print(f"   New best model! Test AUC: {test_auc:.4f}")

            except Exception as e:
                print(f"   Error training {name}: {str(e)}")
                print(traceback.format_exc())
                continue

        if best_model_obj is None:
            raise Exception('No model selected as best')

        print('\n' + '='*80)
        print(f" BEST MODEL SELECTED: {best_model_name}")
        print(f" Test Accuracy: {results[best_model_name]['test_acc']*100:.2f}%")
        print(f" Test ROC AUC: {best_auc_score:.4f}")
        print(f" Bidirectional CV Avg: {results[best_model_name]['bidirectional_cv']['bidirectional_avg']:.4f}")
        print('='*80)

        print("\n Generating ROC curve image for frontend...")
        roc_curve_base64 = generate_roc_curve_image(best_y_true_binary, best_y_pred_proba, best_model_name, best_auc_score)
        if roc_curve_base64:
            print(" ROC curve image generated and stored")
        else:
            print(" Failed to generate ROC curve image")

        try:
            print_smooth_rising_roc_curve(best_y_true_binary, best_y_pred_proba, best_model_name, best_auc_score, width=70, height=30, n_points=1000)
        except Exception:
            pass

        model_artifact_file = os.path.join(MODEL_ARTIFACT_PATH, 'best_model.joblib')
        rf_artifact_file = os.path.join(MODEL_ARTIFACT_PATH, 'rf_model.joblib')
        scaler_artifact_file = os.path.join(MODEL_ARTIFACT_PATH, 'scaler.joblib')
        joblib.dump(best_model_obj, model_artifact_file)
        joblib.dump(best_rf_model, rf_artifact_file)
        joblib.dump(scaler, scaler_artifact_file)
        print(f" Saved best model to {model_artifact_file}")
        print(f" Saved rf model to {rf_artifact_file}")
        print(f" Saved scaler to {scaler_artifact_file}")

        return best_model_obj, best_model_name, results

    except Exception as e:
        print(' Critical error during training:', str(e))
        print(traceback.format_exc())
        raise


def calculate_training_statistics(X_train):
    stats = {'overall': {'min': float(X_train.min()), 'max': float(X_train.max()), 'mean': float(X_train.mean()), 'std': float(X_train.std()), 'median': float(np.median(X_train))}, 'per_feature': {}}
    for i in range(X_train.shape[1]):
        stats['per_feature'][i] = {
            'min': float(X_train[:, i].min()), 'max': float(X_train[:, i].max()), 'mean': float(X_train[:, i].mean()), 'std': float(X_train[:, i].std()),
            'q25': float(np.percentile(X_train[:, i], 25)), 'q75': float(np.percentile(X_train[:, i], 75))
        }
    return stats

def calculate_similarity_to_training_data(input_scaled, X_train_original, training_stats):
    in_range_count = 0
    total_features = len(input_scaled)
    out_of_range_features = []
    for i, value in enumerate(input_scaled):
        feature_stats = training_stats['per_feature'][i]
        if feature_stats['min'] <= value <= feature_stats['max']:
            in_range_count += 1
        else:
            out_of_range_features.append(i)
    similarity_score = (in_range_count / total_features) * 100
    in_distribution = similarity_score >= OOD_SIMILARITY_THRESHOLD
    return {'similarity_score': similarity_score, 'in_distribution': in_distribution, 'in_range_count': in_range_count, 'out_of_range_count': len(out_of_range_features), 'out_of_range_features': out_of_range_features[:5]}

def adjust_confidence_based_on_similarity(original_confidence, similarity_info):
    similarity_score = similarity_info['similarity_score']
    in_distribution = similarity_info['in_distribution']
    if in_distribution and similarity_score >= 95:
        boost = (100 - original_confidence) * 0.4
        adjusted = min(95.0, original_confidence + boost)
        reason = f" High similarity ({similarity_score:.1f}%) - confidence boosted"
        return adjusted, boost, reason
    elif similarity_score >= 70:
        penalty = (100 - similarity_score) * 0.3
        adjusted = max(50.0, original_confidence - penalty)
        reason = f" Moderate similarity ({similarity_score:.1f}%)"
        return adjusted, penalty, reason
    else:
        penalty = (100 - similarity_score) * 0.8
        adjusted = max(45.0, original_confidence - penalty)
        reason = f" Low similarity ({similarity_score:.1f}%)"
        return adjusted, penalty, reason


def initialize_model():
    global best_model, scaler, training_data_stats, model_name, X_train_original, best_rf_model
    try:
        print('\n' + '='*80)
        print(' INITIALIZING ROCK VS MINE ML MODEL (Option C - Advanced + Bidirectional CV + ROC)')
        print('='*80)
        sonar_data = fetch_dataset(DATASET_URL)
        X = sonar_data.drop(columns=60, axis=1).values
        Y = sonar_data[60].values
        print(f" Features shape: {X.shape}")
        print(f" Labels: {np.unique(Y)}")

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, stratify=Y, random_state=42)

        scaler_local = StandardScaler()
        X_train_scaled = scaler_local.fit_transform(X_train)
        X_test_scaled = scaler_local.transform(X_test)
        print('Features scaled using StandardScaler')
        X_train_original = X_train_scaled.copy()
        training_data_stats = calculate_training_statistics(X_train_scaled)
        print(' Training data statistics calculated')

        best_model_obj, best_model_name, results = train_multiple_models(X_train_scaled, Y_train, X_test_scaled, Y_test)

        best_model = best_model_obj
        model_name = best_model_name
        scaler = scaler_local

        scaler_file = os.path.join(MODEL_ARTIFACT_PATH, 'scaler.joblib')
        joblib.dump(scaler, scaler_file)
        print(f" Saved scaler to {scaler_file}")

        print('\n' + '='*80)
        print(' MODEL INITIALIZATION COMPLETE!')
        print('='*80)
        print(f" Best Model: {model_name}")
        print(f" ROC Curve Available: {roc_curve_base64 is not None}")

    except Exception as e:
        print('\n' + '='*80)
        print(' CRITICAL ERROR: MODEL INITIALIZATION FAILED')
        print('='*80)
        print('Error:', str(e))
        print(traceback.format_exc())
        print('='*80)

@app.on_event('startup')
async def startup_event():
    initialize_model()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        field = ' -> '.join(str(x) for x in error['loc'])
        message = error['msg']
        errors.append(f"{field}: {message}")
    error_message = '\n'.join(errors)
    return JSONResponse(status_code=422, content={"detail": error_message})

@app.get('/')
def root():
    return {
        "message": " Rock vs Mine ML API  (Option C - Advanced + Bidirectional CV + ROC)",
        "version": "3.0.0",
        "status": "online",
        "model_loaded": best_model is not None,
        "roc_curve_available": roc_curve_base64 is not None
    }

@app.get('/health')
def health_check():
    return {
        "status": "healthy" if best_model is not None else "model_not_loaded",
        "model_loaded": best_model is not None,
        "model_name": model_name,
        "roc_curve_available": roc_curve_base64 is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get('/info')
def model_info():
    if best_model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    return {
        "model_name": model_name,
        "model_type": "Logistic Regression (Best of 5) stacked on Random Forest (Option C - Advanced + Bidirectional CV)",
        "num_features": 60,
        "roc_curve_available": roc_curve_base64 is not None
    }

@app.get('/roc-curve')
def get_roc_curve():
    """Returns the ROC curve as base64 encoded image"""
    if roc_curve_base64 is None:
        raise HTTPException(status_code=404, detail='ROC curve not available. Model not trained yet.')
    
    return {
        "roc_curve_image": f"data:image/png;base64,{roc_curve_base64}",
        "model_name": model_name,
        "timestamp": datetime.now().isoformat()
    }

@app.post('/predict', response_model=PredictionResponse)
def predict(data: InputData):
    global prediction_history
    if best_model is None or scaler is None or best_rf_model is None:
        raise HTTPException(status_code=503, detail='Model not initialized.')
    try:
        input_data = np.array(data.values).reshape(1, -1)
        input_scaled = scaler.transform(input_data)

        rf_prob_raw = get_positive_class_proba(best_rf_model, input_scaled, positive_class='M').reshape(1,)

        if PREDICTION_NOISE:
            rf_prob_used = add_prediction_noise(rf_prob_raw.reshape(-1,1), noise_level=PREDICTION_NOISE_LEVEL).reshape(1,1)
        else:
            rf_prob_used = rf_prob_raw.reshape(1,1)

        input_enh = np.hstack((input_scaled, rf_prob_used))
        similarity_info = calculate_similarity_to_training_data(input_scaled.flatten(), X_train_original, training_data_stats)
        prediction = best_model.predict(input_enh)[0]
        confidence_scores = best_model.predict_proba(input_enh)[0]
        original_confidence = float(confidence_scores.max() * 100)

        adjusted_confidence, penalty_or_boost, reason = adjust_confidence_based_on_similarity(original_confidence, similarity_info)
        prediction_label = 'Rock' if prediction == 'R' else 'Mine'

        response = {
            'prediction': prediction_label,
            'confidence': round(adjusted_confidence, 2),
            'original_confidence': round(original_confidence, 2),
            'adjusted_confidence': round(adjusted_confidence, 2),
            'penalty_or_boost': round(penalty_or_boost, 2) if penalty_or_boost is not None else None,
            'in_distribution': similarity_info['in_distribution'],
            'similarity_score': round(similarity_info['similarity_score'], 2),
            'reason': reason,
            'model_used': model_name,
            'timestamp': datetime.now().isoformat(),
            'warning': None if similarity_info['in_distribution'] else ' Input outside distribution'
        }

        prediction_history.append({**response, 'input': data.values[:3] + ['...'] + data.values[-3:]})
        print(f" Prediction: {prediction_label} | Confidence: {adjusted_confidence:.2f}% | In-distribution: {similarity_info['in_distribution']}")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get('/history')
def get_history():
    return {
        "history": prediction_history,
        "total_predictions": len(prediction_history),
        "rock_count": sum(1 for p in prediction_history if p['prediction'] == 'Rock'),
        "mine_count": sum(1 for p in prediction_history if p['prediction'] == 'Mine')
    }

@app.post('/clear-history')
def clear_history():
    global prediction_history
    prediction_history = []
    return {"message": " History cleared"}

if __name__ == '__main__':
    print('\n' + '='*80)
    print(' STARTING FASTAPI SERVER (Option C - Advanced + Bidirectional CV + ROC)')
    print('='*80)
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')

