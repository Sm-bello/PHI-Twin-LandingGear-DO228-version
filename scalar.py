import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

from aerotwin_config import DATASET_PATH, SCALER_SCALAR_PATH, SCALER_SEQ_PATH

# 1. LOAD DATASET (same as before)
data_path = DATASET_PATH
df = pd.read_csv(data_path)

# 2. FEATURE ENGINEERING (same)
seq_cols = df.columns[23:273]
scalar_cols = ['Mass_kg', 'SinkRate_ms', 'Temp_C', 'Friction_Coef']

X_seq = df[seq_cols].values.reshape(-1, 5, 50).transpose(0, 2, 1)  # (samples, 50, 5)
X_scalar = df[scalar_cols].values

# 3. FIT & SAVE SCALERS ONLY — no model training
scaler_seq = StandardScaler()
X_seq_flat = X_seq.reshape(-1, 5)
scaler_seq.fit(X_seq_flat)                     # <-- fit only
joblib.dump(scaler_seq, SCALER_SEQ_PATH)

scaler_scalar = StandardScaler()
scaler_scalar.fit(X_scalar)                    # <-- fit only
joblib.dump(scaler_scalar, SCALER_SCALAR_PATH)

print("✅ Scalers fitted and saved successfully!")
print("You can now use these in your mission script without retraining.")