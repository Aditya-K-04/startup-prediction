import joblib
BASE = r"C:\Users\chand\OneDrive\Desktop\7th sem\startup-prediction"
scaler = joblib.load(BASE + r"\models\optimized_scaler.pkl")
print("Total columns:", len(scaler.feature_names_in_))
print("\nAll columns:")
for i, f in enumerate(scaler.feature_names_in_):
    print(i, f)