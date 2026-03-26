# -*- coding: utf-8 -*-
"""
Run this ONCE - saves encoder as JSON instead of pickle.
"""
import os, json, joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split

BASE    = r"C:\Users\chand\OneDrive\Desktop\7th sem\startup-prediction"
MDL_DIR = os.path.join(BASE, "models")
DATA    = os.path.join(BASE, "data", "integrated", "final_training_data.csv")

# Load old pickle with a dummy class to extract the data
import pickle

class TargetEncoder:  # dummy - just to unpickle
    pass

with open(os.path.join(MDL_DIR, "target_encoder.pkl"), "rb") as f:
    old = pickle.load(f)

# Extract the encoding dictionaries from the old object
encoder_data = {
    "global_mean": float(old.global_mean),
    "smoothing":   int(old.smoothing),
    "encodings":   {k: {str(kk): float(vv) for kk, vv in v.items()}
                    for k, v in old.encodings.items()}
}

out_path = os.path.join(MDL_DIR, "target_encoder.json")
with open(out_path, "w") as f:
    json.dump(encoder_data, f)

print("Saved JSON encoder to:", out_path)
print("Columns encoded:", list(encoder_data["encodings"].keys()))
print("Done! Restart uvicorn now.")