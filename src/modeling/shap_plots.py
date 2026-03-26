# -*- coding: utf-8 -*-
"""
shap_plots.py
Generates SHAP explainability plots and saves to results/
Run AFTER train_model.py  (needs saved model + scaler)
pip install shap  (if not installed)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

BASE    = r"C:\Users\chand\OneDrive\Desktop\7th sem\startup-prediction"
DATA    = os.path.join(BASE, "data", "integrated", "final_training_data.csv")
MDL_DIR = os.path.join(BASE, "models")
RES_DIR = os.path.join(BASE, "results")
os.makedirs(RES_DIR, exist_ok=True)

plt.rcParams.update({'figure.dpi': 150, 'font.size': 10})


# =============================================
# TARGET ENCODER  (must match train_model.py)
# =============================================
class TargetEncoder:
    def __init__(self, smoothing=10):
        self.smoothing = smoothing
        self.global_mean = 0.5
        self.encodings = {}

    def fit(self, df, y, cols):
        self.global_mean = y.mean()
        for col in cols:
            if col not in df.columns:
                continue
            tmp   = pd.DataFrame({'x': df[col], 'y': y.values})
            stats = tmp.groupby('x')['y'].agg(['mean', 'count'])
            enc   = ((stats['mean'] * stats['count'] + self.global_mean * self.smoothing)
                     / (stats['count'] + self.smoothing))
            self.encodings[col] = enc.to_dict()

    def transform(self, df, cols):
        out = df.copy()
        for col in cols:
            if col in self.encodings:
                out[col + '_te2'] = df[col].map(self.encodings[col]).fillna(self.global_mean)
        return out


# =============================================
# LOAD
# =============================================
def load_data():
    print("Loading data and model...")
    model  = joblib.load(os.path.join(MDL_DIR, "optimized_model.pkl"))
    scaler = joblib.load(os.path.join(MDL_DIR, "optimized_scaler.pkl"))

    df = pd.read_csv(DATA, low_memory=False)
    y  = df['success']
    X  = df.drop(columns=['success'])
    X  = X.drop(columns=X.select_dtypes(include=['object','category']).columns)

    _, X_te, _, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    te_path = os.path.join(MDL_DIR, "target_encoder.pkl")
    if os.path.exists(te_path):
        te     = joblib.load(te_path)
        te_cols= [c for c in ['main_cat_code','sub_cat_code','country_code']
                  if c in X_te.columns]
        X_te   = te.transform(X_te, te_cols)

    feat_names  = X_te.columns.tolist()
    X_te_s      = scaler.transform(X_te.fillna(0))

    # Use XGBoost sub-model for SHAP (much faster than stacking)
    xgb_model   = model.named_estimators_['xgb']

    # Sample 2000 rows for speed
    np.random.seed(42)
    idx         = np.random.choice(len(X_te_s), size=min(2000, len(X_te_s)), replace=False)
    X_sample    = X_te_s[idx]
    y_sample    = y_te.values[idx]

    print("  Sample size for SHAP:", len(X_sample))
    print("  Features:", len(feat_names))
    return xgb_model, X_sample, y_sample, feat_names, X_te_s, y_te.values


# =============================================
# PLOT 1: SHAP SUMMARY (beeswarm)
# =============================================
def plot_shap_summary(explainer, shap_values, feat_names, X_sample):
    print("Plotting SHAP summary (beeswarm)...")
    fig, ax = plt.subplots(figsize=(10, 10))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feat_names,
        max_display=20,
        show=False,
        plot_size=None
    )
    plt.title('SHAP Summary Plot - Top 20 Features', fontweight='bold', pad=15)
    plt.tight_layout()
    path = os.path.join(RES_DIR, 'shap_summary.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved:", path)


# =============================================
# PLOT 2: SHAP BAR (mean absolute)
# =============================================
def plot_shap_bar(explainer, shap_values, feat_names, X_sample):
    print("Plotting SHAP bar (mean |SHAP|)...")
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feat_names,
        plot_type='bar',
        max_display=20,
        show=False
    )
    plt.title('Mean |SHAP Value| - Feature Importance', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(RES_DIR, 'shap_bar.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved:", path)


# =============================================
# PLOT 3: SHAP WATERFALL (single predictions)
# =============================================
def plot_shap_waterfall(explainer, X_sample, y_sample, feat_names, clean_model):
    print("Plotting SHAP waterfall (best Success + best Failed)...")

    shap_vals_all = explainer(X_sample)

    # Find most confident correct Success and Failed predictions
    # Get probabilities directly from the clean_model (not explainer.model)
    proba = clean_model.predict_proba(X_sample)[:, 1]

    success_mask = (y_sample == 1)
    failed_mask  = (y_sample == 0)

    best_success_idx = int(np.where(success_mask)[0][np.argmax(proba[success_mask])])
    best_failed_idx  = int(np.where(failed_mask)[0][np.argmin(proba[failed_mask])])

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, idx, title in [
        (axes[0], best_success_idx, 'Most Confident SUCCESS Prediction'),
        (axes[1], best_failed_idx,  'Most Confident FAILED Prediction'),
    ]:
        plt.sca(ax)
        shap.waterfall_plot(
            shap.Explanation(
                values       = shap_vals_all.values[idx],
                base_values  = shap_vals_all.base_values[idx],
                data         = X_sample[idx],
                feature_names= feat_names
            ),
            max_display=15,
            show=False
        )
        ax.set_title(title, fontweight='bold', pad=10)

    plt.tight_layout()
    path = os.path.join(RES_DIR, 'shap_waterfall.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved:", path)


# =============================================
# PLOT 4: SHAP DEPENDENCE  (top 2 features)
# =============================================
def plot_shap_dependence(shap_values, X_sample, feat_names):
    print("Plotting SHAP dependence plots...")

    # Top 2 features by mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    top2_idx = np.argsort(mean_abs)[::-1][:2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, feat_idx in zip(axes, top2_idx):
        feat = feat_names[feat_idx]
        # Find best interaction feature
        shap.dependence_plot(
            feat_idx, shap_values, X_sample,
            feature_names=feat_names,
            ax=ax, show=False,
            interaction_index='auto'
        )
        ax.set_title('SHAP Dependence: %s' % feat, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(RES_DIR, 'shap_dependence.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved:", path)


# =============================================
# PLOT 5: SHAP DECISION PLOT (10 examples)
# =============================================
def plot_shap_decision(explainer, shap_values, X_sample, y_sample, feat_names):
    print("Plotting SHAP decision plot...")

    # Pick 5 successes + 5 failures
    s_idx = np.where(y_sample == 1)[0][:5]
    f_idx = np.where(y_sample == 0)[0][:5]
    idx   = np.concatenate([s_idx, f_idx])
    labels= ['Success']*5 + ['Failed']*5

    fig, ax = plt.subplots(figsize=(10, 10))
    shap.decision_plot(
        explainer.expected_value,
        shap_values[idx],
        feature_names=feat_names,
        feature_display_range=slice(-1, -21, -1),
        show=False,
        highlight=range(5),   # highlight success cases
        legend_labels=labels,
        legend_location='lower right'
    )
    plt.title('SHAP Decision Plot (5 Success vs 5 Failed)', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(RES_DIR, 'shap_decision.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved:", path)


# =============================================
# RUN ALL
# =============================================
if __name__ == "__main__":
    print("="*55)
    print("  SHAP EXPLAINABILITY PLOTS")
    print("="*55)

    xgb_model, X_sample, y_sample, feat_names, X_te_s, y_te = load_data()

    print("\nComputing SHAP values (TreeExplainer - fast)...")
    # Fix for SHAP/XGBoost version mismatch (base_score string bug):
    # save model to JSON and reload so base_score is parsed correctly
    import tempfile, os as _os
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name
    xgb_model.save_model(tmp_path)
    import xgboost as _xgb
    clean_model = _xgb.XGBClassifier()
    clean_model.load_model(tmp_path)
    _os.unlink(tmp_path)

    explainer   = shap.TreeExplainer(clean_model)
    shap_values = explainer.shap_values(X_sample)

    print("SHAP values shape:", shap_values.shape)

    plot_shap_summary(explainer, shap_values, feat_names, X_sample)
    plot_shap_bar(explainer, shap_values, feat_names, X_sample)
    plot_shap_waterfall(explainer, X_sample, y_sample, feat_names, clean_model)
    plot_shap_dependence(shap_values, X_sample, feat_names)
    plot_shap_decision(explainer, shap_values, X_sample, y_sample, feat_names)

    print("\n[DONE] All SHAP plots saved to:", RES_DIR)
    print("\nFiles generated:")
    for f in ['shap_summary.png', 'shap_bar.png', 'shap_waterfall.png',
              'shap_dependence.png', 'shap_decision.png']:
        print(" ", f)