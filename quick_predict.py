#!/usr/bin/env python3
"""
Quick script to predict with saved model
Usage: python quick_predict.py model_file.pkl data_file.csv
"""

import sys
import pickle
import pandas as pd

if len(sys.argv) != 3:
    print("Usage: python quick_predict.py <model_file.pkl> <data_file.csv>")
    sys.exit(1)

model_file = sys.argv[1]
data_file = sys.argv[2]

# Load model
with open(model_file, 'rb') as f:
    saved_data = pickle.load(f)

model = saved_data['model']
scaler = saved_data['scaler']
label_encoder = saved_data['label_encoder']
feature_columns = saved_data['feature_columns']

# Load and predict
data = pd.read_csv(data_file)
X = data[feature_columns].values
X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)

if label_encoder:
    predictions = label_encoder.inverse_transform(predictions)

# Print results
for i, pred in enumerate(predictions):
    print(f"Row {i+1}: {pred}")

print(f"\nTotal predictions: {len(predictions)}")
print(f"Unique predictions: {set(predictions)}")