"""
AeroTwin V16 Model Evaluation – Corrected
------------------------------------------
Uses the original train/validation/test split from train_val_split_V16p1.mat
and computes proper metrics.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import scipy.io

from aerotwin_config import (
    BASE_DIR as _BASE_PATH,
    DATASET_PATH,
    FIGURES_DIR,
    MODEL_PATH,
    SCALER_SCALAR_PATH,
    SCALER_SEQ_PATH,
    SPLIT_MAT_PATH,
)

# --------------------------- Configuration -----------------------------
BASE_DIR = str(_BASE_PATH)
DATA_PATH = DATASET_PATH
SCALER_SEQ_PATH = SCALER_SEQ_PATH
SCALER_SCALAR_PATH = SCALER_SCALAR_PATH
SPLIT_PATH = SPLIT_MAT_PATH

OUTPUT_DIR = FIGURES_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ['Normal', 'N2 Leak', 'Worn Seal', 'Early Degrad', 'Thermal Degrad',
               'Brake Fade', 'Tire Burst', 'Corrosion', 'Hard Landing',
               'Combined Fault', 'Impending Failure']

# --------------------------- Load Data ---------------------------------
print("📊 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   Total samples: {len(df)}")

# Extract features
seq_cols = df.columns[23:273]
X_seq = df[seq_cols].values.reshape(-1, 5, 50).transpose(0, 2, 1)   # (samples, 50, 5)

scalar_cols = ['Mass_kg', 'SinkRate_ms', 'Temp_C', 'Friction_Coef']
X_scalar = df[scalar_cols].values

y_true = df['Class'].values
y_rul_true = df['RUL_Percent'].values

print(f"   X_seq shape: {X_seq.shape}")
print(f"   X_scalar shape: {X_scalar.shape}")

# --------------------------- Load Scalers ------------------------------
print("\n📦 Loading scalers...")
scaler_seq = joblib.load(SCALER_SEQ_PATH)
scaler_scalar = joblib.load(SCALER_SCALAR_PATH)

X_seq_scaled = X_seq.reshape(-1, 5)
X_seq_scaled = scaler_seq.transform(X_seq_scaled).reshape(-1, 50, 5)
X_scalar_scaled = scaler_scalar.transform(X_scalar)

# --------------------------- Load Original Split -----------------------
print("\n📂 Loading original train/val/test split from MATLAB...")
mat = scipy.io.loadmat(SPLIT_PATH)
sp = mat['sp']   # structured array
# Access fields (they are 1x1 arrays containing the index arrays)
train_idx = sp['train_idx'][0,0].flatten() - 1   # convert to 0‑based
val_idx   = sp['val_idx'][0,0].flatten() - 1
test_idx  = sp['test_idx'][0,0].flatten() - 1

print(f"   Train indices: {len(train_idx)}")
print(f"   Val indices:   {len(val_idx)}")
print(f"   Test indices:  {len(test_idx)}")

# Use only the test set for final evaluation
X_seq_test   = X_seq_scaled[test_idx]
X_scal_test  = X_scalar_scaled[test_idx]
y_test       = y_true[test_idx]
y_rul_test   = y_rul_true[test_idx]

print(f"\n📈 Test samples: {len(y_test)}")

# --------------------------- Load Model --------------------------------
print("\n🤖 Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("   Model loaded.")

# --------------------------- Classification Evaluation -----------------
print("\n" + "="*60)
print("CLASSIFICATION PERFORMANCE (ORIGINAL TEST SET)")
print("="*60)

predictions = model.predict([X_seq_test, X_scal_test], verbose=1)
y_pred = np.argmax(predictions, axis=1)

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred,
                                                    labels=range(11),
                                                    average=None)
macro_f1 = np.mean(f1)

print(f"\nOverall Accuracy: {acc*100:.2f}%")
print(f"Macro F1-Score:   {macro_f1:.4f}\n")

print("Per‑Class Metrics:")
print("   Class               Precision  Recall    F1-Score")
for i, name in enumerate(CLASS_NAMES):
    print(f"   {i:2d} {name:<18} {prec[i]:.3f}     {rec[i]:.3f}     {f1[i]:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            annot_kws={'size': 10})
plt.title(f'Confusion Matrix – Original Test Set (Acc = {acc*100:.2f}%)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'evaluation_confusion_matrix_original.png'), dpi=300)
print("\n✅ Confusion matrix saved to evaluation_confusion_matrix_original.png")

# --------------------------- RUL Prediction Evaluation -----------------
print("\n" + "="*60)
print("RUL PREDICTION PERFORMANCE (ORIGINAL TEST SET)")
print("="*60)

# Extract features from the second‑to‑last layer, using ONLY the training set
# to train the Random Forest (avoid leakage).
feature_extractor = Model(inputs=model.inputs,
                          outputs=model.layers[-2].output)

print("\n🔄 Extracting features for training and test sets...")
features_all = feature_extractor.predict([X_seq_scaled, X_scalar_scaled], verbose=1)

feat_train = features_all[train_idx]
feat_test  = features_all[test_idx]
y_rul_train = y_rul_true[train_idx]

print(f"\n🌲 Training Random Forest regressor on training features ({len(feat_train)} samples)...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(feat_train, y_rul_train)
y_rul_pred = rf.predict(feat_test)

# Compute metrics
mae = mean_absolute_error(y_rul_test, y_rul_pred)
rmse = np.sqrt(mean_squared_error(y_rul_test, y_rul_pred))
r2 = r2_score(y_rul_test, y_rul_pred)

print(f"\nRUL Prediction Metrics:")
print(f"   MAE  = {mae:.2f}%")
print(f"   RMSE = {rmse:.2f}%")
print(f"   R²   = {r2:.4f}")

# Scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(y_rul_test, y_rul_pred, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
plt.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual RUL (%)')
plt.ylabel('Predicted RUL (%)')
plt.title(f'RUL Prediction (MAE = {mae:.2f}%, R² = {r2:.4f})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'evaluation_rul_scatter_original.png'), dpi=300)
print("\n✅ RUL scatter plot saved to evaluation_rul_scatter_original.png")

print("\n✅ Corrected evaluation complete. Numbers now reflect the original test set.")