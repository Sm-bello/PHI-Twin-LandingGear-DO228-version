import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

from aerotwin_config import (
    DATASET_PATH,
    MODEL_PATH,
    SCALER_SCALAR_PATH,
    SCALER_SEQ_PATH,
)

# 1. LOAD DATASET
data_path = DATASET_PATH
df = pd.read_csv(data_path)

# 2. FEATURE ENGINEERING
# Sequence columns (50 timesteps * 5 sensors)
# Signals: Defl(24-73) | Velo(74-123) | Force(124-173) | Accel(174-223) | Pressure(224-273)
seq_cols = df.columns[23:273]
scalar_cols = ['Mass_kg', 'SinkRate_ms', 'Temp_C', 'Friction_Coef']

X_seq = df[seq_cols].values.reshape(-1, 5, 50).transpose(0, 2, 1) # Reshape to (Samples, 50, 5)
X_scalar = df[scalar_cols].values
y = df['Class'].values

# 3. PREPROCESSING
scaler_seq = StandardScaler()
X_seq_scaled = X_seq.reshape(-1, 5)
X_seq_scaled = scaler_seq.fit_transform(X_seq_scaled).reshape(-1, 50, 5)

scaler_scalar = StandardScaler()
X_scalar_scaled = scaler_scalar.fit_transform(X_scalar)

# 4. DUAL-INPUT MODEL DEFINITION
def build_robust_model():
    # Branch 1: The Sequence Processor (CNN + LSTM)
    seq_input = layers.Input(shape=(50, 5), name="Sensor_Signals")
    x = layers.Conv1D(64, kernel_size=3, activation='relu')(seq_input)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)

    # Branch 2: The Context Processor (Metadata)
    scalar_input = layers.Input(shape=(4,), name="Flight_Context")
    y_branch = layers.Dense(32, activation='relu')(scalar_input)

    # Fusion Layer
    combined = layers.concatenate([x, y_branch])
    z = layers.Dense(64, activation='relu')(combined)
    output = layers.Dense(11, activation='softmax', name="Fault_Classifier")(z)

    model = models.Model(inputs=[seq_input, scalar_input], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 5. EXECUTE TRAINING
model = build_robust_model()
X_train_seq, X_test_seq, X_train_scal, X_test_scal, y_train, y_test = train_test_split(
    X_seq_scaled, X_scalar_scaled, y, test_size=0.2, stratify=y, random_state=42
)

history = model.fit(
    [X_train_seq, X_train_scal], y_train,
    validation_data=([X_test_seq, X_test_scal], y_test),
    epochs=50, batch_size=64,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

# 6. EVALUATE & SAVE
model.save(MODEL_PATH)
joblib.dump(scaler_seq, SCALER_SEQ_PATH)
joblib.dump(scaler_scalar, SCALER_SCALAR_PATH)
print(f"🚀 Training Complete! Model saved to {MODEL_PATH}")