
"""
train_model.py  -  Reads final_training_data.csv (output of data_integrator.py)
Expected: 100+ features, ~72-74% accuracy, ~0.78 ROC-AUC
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                              accuracy_score, precision_recall_curve)
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE    = r"C:\Users\chand\OneDrive\Desktop\7th sem\startup-prediction"
DATA    = os.path.join(BASE, "data", "integrated", "final_training_data.csv")
MDL_DIR = os.path.join(BASE, "models")
RES_DIR = os.path.join(BASE, "results")
for d in [MDL_DIR, RES_DIR]:
    os.makedirs(d, exist_ok=True)


# =============================================
# TARGET ENCODER  (leak-free, fit on train only)
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
# MODELS
# =============================================
def get_xgb():
    return xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, colsample_bylevel=0.7,
        min_child_weight=5, gamma=0.1,
        reg_alpha=0.3, reg_lambda=2.0,
        scale_pos_weight=1.2,
        eval_metric='logloss', random_state=42, n_jobs=-1
    )

def get_lgb():
    return lgb.LGBMClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.7,
        min_child_samples=20, reg_alpha=0.3, reg_lambda=2.0,
        class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
    )

def get_catboost():
    return CatBoostClassifier(
        iterations=500, depth=5, learning_rate=0.05,
        l2_leaf_reg=5, border_count=128,
        random_seed=42, verbose=0, auto_class_weights='Balanced'
    )

def get_rf():
    return RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1
    )

def build_stacking():
    return StackingClassifier(
        estimators=[
            ('xgb', get_xgb()), ('lgb', get_lgb()),
            ('cat', get_catboost()), ('rf', get_rf())
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        cv=5, stack_method='predict_proba', n_jobs=-1, passthrough=True
    )


# =============================================
# MAIN
# =============================================
def run():
    print("=" * 55)
    print("  KICKSTARTER SUCCESS PREDICTOR")
    print("=" * 55)

    # --- Load integrated data ---
    print("Loading", DATA)
    df = pd.read_csv(DATA, low_memory=False)
    print("Shape:", df.shape, "| Success: %.1f%%" % (df['success'].mean()*100))

    y = df['success']
    X = df.drop(columns=['success'])
    X = X.fillna(0)

    print("Features:", X.shape[1])

    # --- Train/test split ---
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Leak-free target encoding on any remaining code columns ---
    te_cols = [c for c in ['main_cat_code', 'sub_cat_code', 'country_code']
               if c in X_tr.columns]
    if te_cols:
        te = TargetEncoder(smoothing=10)
        te.fit(X_tr, y_tr, te_cols)
        X_tr = te.transform(X_tr, te_cols)
        X_te = te.transform(X_te, te_cols)
        X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
        joblib.dump(te, os.path.join(MDL_DIR, 'target_encoder.pkl'))

    # --- Drop any remaining string/object columns ---
    obj_cols = X_tr.select_dtypes(include=['object', 'category']).columns.tolist()
    if obj_cols:
        print("Dropping non-numeric columns:", obj_cols)
        X_tr = X_tr.drop(columns=obj_cols)
        X_te = X_te.drop(columns=[c for c in obj_cols if c in X_te.columns])

    feature_names = X_tr.columns.tolist()
    print("Final feature count:", len(feature_names))

    # --- Scale ---
    scaler  = RobustScaler()
    X_tr_s  = scaler.fit_transform(X_tr.fillna(0))
    X_te_s  = scaler.transform(X_te.fillna(0))

    # --- CV check ---
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(get_xgb(), X_tr_s, y_tr, cv=cv,
                             scoring='roc_auc', n_jobs=-1)
    print("[CV] XGBoost ROC-AUC: %.4f +/- %.4f" % (scores.mean(), scores.std()))

    # --- Train stacking ensemble ---
    print("\nFitting Stacking Ensemble (XGB + LGB + CatBoost + RF)...")
    model = build_stacking()
    model.fit(X_tr_s, y_tr)

    y_pred  = model.predict(X_te_s)
    y_proba = model.predict_proba(X_te_s)[:, 1]

    tr_acc = accuracy_score(y_tr, model.predict(X_tr_s))
    te_acc = accuracy_score(y_te, y_pred)
    te_auc = roc_auc_score(y_te, y_proba)

    print("\nTrain Acc: %.1f%%  |  Test Acc: %.1f%%  |  ROC-AUC: %.4f"
          % (tr_acc*100, te_acc*100, te_auc))

    # --- Optimal threshold ---
    precisions, recalls, thresholds = precision_recall_curve(y_te, y_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_thresh = float(thresholds[np.argmax(f1s)])
    y_pred_tuned = (y_proba >= best_thresh).astype(int)

    print("\nCLASSIFICATION REPORT (threshold=0.5):")
    print(classification_report(y_te, y_pred, target_names=['Failed', 'Success']))

    print("CLASSIFICATION REPORT (tuned threshold=%.3f):" % best_thresh)
    print(classification_report(y_te, y_pred_tuned, target_names=['Failed', 'Success']))

    # --- Business impact ---
    top20_thresh = np.percentile(y_proba, 80)
    top20_acc    = y_te[y_proba >= top20_thresh].mean()
    baseline     = y.mean()
    print("BUSINESS IMPACT:")
    print("  Top 20%% predicted -> %.1f%% success  (baseline %.1f%%)"
          % (top20_acc*100, baseline*100))
    print("  Lift: %.2fx" % (top20_acc / baseline))

    # --- Feature importance ---
    imp = model.named_estimators_['xgb'].feature_importances_
    fi  = (pd.DataFrame({'feature': feature_names, 'importance': imp})
             .sort_values('importance', ascending=False).head(20))
    plt.figure(figsize=(10, 10))
    sns.barplot(data=fi, y='feature', x='importance', palette='viridis')
    plt.title('Top 20 Feature Importances', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, 'feature_importance.png'), dpi=300)
    plt.close()
    print("\nTop 8 features:", fi['feature'].tolist()[:8])

    # --- Save ---
    joblib.dump(model,  os.path.join(MDL_DIR, 'optimized_model.pkl'))
    joblib.dump(scaler, os.path.join(MDL_DIR, 'optimized_scaler.pkl'))

    print("\n[DONE] Accuracy: %.1f%%  |  ROC-AUC: %.4f" % (te_acc*100, te_auc))
    return te_acc, te_auc


if __name__ == "__main__":
    run()