
"""
model_eval.py
Generates all evaluation plots and saves them to results/
Run AFTER train_model.py has saved the model + scaler
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report,
    roc_auc_score, accuracy_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

BASE    = r"C:\Users\chand\OneDrive\Desktop\7th sem\startup-prediction"
DATA    = os.path.join(BASE, "data", "integrated", "final_training_data.csv")
MDL_DIR = os.path.join(BASE, "models")
RES_DIR = os.path.join(BASE, "results")
os.makedirs(RES_DIR, exist_ok=True)

# Plot style
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
})
PALETTE = {'Failed': '#E74C3C', 'Success': '#2ECC71'}


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



def load_everything():
    print("Loading model and data...")
    model  = joblib.load(os.path.join(MDL_DIR, "optimized_model.pkl"))
    scaler = joblib.load(os.path.join(MDL_DIR, "optimized_scaler.pkl"))

    df = pd.read_csv(DATA, low_memory=False)
    y  = df['success']
    X  = df.drop(columns=['success'])

    # Drop object columns
    obj_cols = X.select_dtypes(include=['object', 'category']).columns
    X = X.drop(columns=obj_cols)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Load target encoder if exists
    te_path = os.path.join(MDL_DIR, "target_encoder.pkl")
    if os.path.exists(te_path):
        te = joblib.load(te_path)
        te_cols = [c for c in ['main_cat_code', 'sub_cat_code', 'country_code']
                   if c in X_tr.columns]
        X_tr = te.transform(X_tr, te_cols)
        X_te = te.transform(X_te, te_cols)
        X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)

    X_tr_s = scaler.transform(X_tr.fillna(0))
    X_te_s = scaler.transform(X_te.fillna(0))

    y_proba = model.predict_proba(X_te_s)[:, 1]
    y_pred  = model.predict(X_te_s)

    print("  Test size:", len(y_te), "| ROC-AUC: %.4f" % roc_auc_score(y_te, y_proba))
    return model, scaler, X_te_s, y_te, y_pred, y_proba, X_tr.columns.tolist()


def plot_confusion_matrix(y_te, y_pred, y_proba):
    # Default threshold
    cm1 = confusion_matrix(y_te, y_pred)
    # Tuned threshold
    from sklearn.metrics import precision_recall_curve
    p, r, thresh = precision_recall_curve(y_te, y_proba)
    f1s = 2*p*r/(p+r+1e-8)
    best_t = float(thresh[np.argmax(f1s)])
    y_tuned = (y_proba >= best_t).astype(int)
    cm2 = confusion_matrix(y_te, y_tuned)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cm, title in zip(axes,
                              [cm1, cm2],
                              ['Confusion Matrix (threshold=0.50)',
                               'Confusion Matrix (tuned threshold=%.3f)' % best_t]):
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Failed', 'Success'],
                    yticklabels=['Failed', 'Success'],
                    linewidths=0.5, linecolor='white')
        # Add text with count + percentage
        for i in range(2):
            for j in range(2):
                ax.text(j+0.5, i+0.5,
                        '%d\n(%.1f%%)' % (cm[i,j], cm_pct[i,j]),
                        ha='center', va='center', fontsize=12,
                        color='white' if cm_pct[i,j] > 50 else 'black',
                        fontweight='bold')
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    path = os.path.join(RES_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved:", path)


def plot_roc_curve(y_te, y_proba):
    fpr, tpr, _ = roc_curve(y_te, y_proba)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#3498DB', lw=2.5,
            label='ROC Curve (AUC = %.4f)' % roc_auc)
    ax.fill_between(fpr, tpr, alpha=0.08, color='#3498DB')
    ax.plot([0,1],[0,1], 'k--', lw=1.5, label='Random Classifier (AUC = 0.50)')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Kickstarter Success Prediction')
    ax.legend(loc='lower right', fontsize=10)

    # Annotate best threshold point
    from sklearn.metrics import precision_recall_curve
    p, r, thresh = precision_recall_curve(y_te, y_proba)
    f1s = 2*p*r/(p+r+1e-8)
    best_t = float(thresh[np.argmax(f1s)])
    fpr2, tpr2, thresholds2 = roc_curve(y_te, y_proba)
    idx = np.argmin(np.abs(thresholds2 - best_t))
    ax.scatter(fpr2[idx], tpr2[idx], s=120, zorder=5,
               color='#E74C3C', label='Optimal threshold (%.3f)' % best_t)
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    path = os.path.join(RES_DIR, 'roc_curve.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved:", path)



def plot_pr_curve(y_te, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_te, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx  = np.argmax(f1_scores)
    best_t    = float(thresholds[best_idx])
    pr_auc    = auc(recall, precision)
    baseline  = y_te.mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color='#9B59B6', lw=2.5,
            label='PR Curve (AUC = %.4f)' % pr_auc)
    ax.fill_between(recall, precision, alpha=0.08, color='#9B59B6')
    ax.axhline(baseline, color='gray', lw=1.5, linestyle='--',
               label='Baseline (%.1f%%)' % (baseline*100))
    ax.scatter(recall[best_idx], precision[best_idx], s=120,
               color='#E74C3C', zorder=5,
               label='Best F1=%.3f at t=%.3f' % (f1_scores[best_idx], best_t))
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    path = os.path.join(RES_DIR, 'pr_curve.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved:", path)



def plot_threshold_analysis(y_te, y_proba):
    thresholds = np.linspace(0.1, 0.9, 100)
    accs, precs, recs, f1s = [], [], [], []

    for t in thresholds:
        yp = (y_proba >= t).astype(int)
        accs.append(accuracy_score(y_te, yp))
        precs.append(float(np.nan_to_num(
            confusion_matrix(y_te, yp)[1,1] /
            (confusion_matrix(y_te, yp)[:,1].sum() + 1e-8)
        )))
        recs.append(float(np.nan_to_num(
            confusion_matrix(y_te, yp)[1,1] /
            (confusion_matrix(y_te, yp)[1,:].sum() + 1e-8)
        )))
        f1s.append(f1_score(y_te, yp, zero_division=0))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, accs,  label='Accuracy',  lw=2,   color='#3498DB')
    ax.plot(thresholds, precs, label='Precision', lw=2,   color='#E74C3C')
    ax.plot(thresholds, recs,  label='Recall',    lw=2,   color='#2ECC71')
    ax.plot(thresholds, f1s,   label='F1 Score',  lw=2.5, color='#9B59B6', linestyle='--')
    ax.axvline(0.5,  color='gray',  lw=1, linestyle=':',  label='Default (0.50)')
    best_t = thresholds[np.argmax(f1s)]
    ax.axvline(best_t, color='#E67E22', lw=1.5, linestyle='--',
               label='Best F1 threshold (%.3f)' % best_t)
    ax.set_xlabel('Classification Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Metrics vs Classification Threshold')
    ax.legend(fontsize=10)
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    path = os.path.join(RES_DIR, 'threshold_analysis.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved:", path)


def plot_feature_importance(model, feature_names):
    imp = model.named_estimators_['xgb'].feature_importances_
    fi  = pd.DataFrame({'feature': feature_names, 'importance': imp})
    fi  = fi.sort_values('importance', ascending=False).head(20)

    # Color by feature group
    def get_color(name):
        if 'goal' in name:   return '#3498DB'
        if 'cat'  in name:   return '#9B59B6'
        if 'sub'  in name:   return '#8E44AD'
        if 'dur'  in name:   return '#2ECC71'
        if 'name' in name:   return '#E67E22'
        if 'launch' in name: return '#1ABC9C'
        if 'reddit' in name: return '#E74C3C'
        return '#95A5A6'

    colors = [get_color(f) for f in fi['feature']]

    fig, ax = plt.subplots(figsize=(10, 9))
    bars = ax.barh(fi['feature'], fi['importance'], color=colors, edgecolor='white')
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (XGBoost)')
    ax.set_title('Top 20 Feature Importances by Group')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498DB', label='Goal features'),
        Patch(facecolor='#9B59B6', label='Category features'),
        Patch(facecolor='#2ECC71', label='Duration features'),
        Patch(facecolor='#E67E22', label='Name features'),
        Patch(facecolor='#1ABC9C', label='Launch timing'),
        Patch(facecolor='#E74C3C', label='Reddit features'),
        Patch(facecolor='#95A5A6', label='Other'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    # Value labels
    for bar, val in zip(bars, fi['importance']):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
                '%.4f' % val, va='center', fontsize=8)

    plt.tight_layout()
    path = os.path.join(RES_DIR, 'feature_importance.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved:", path)



def plot_score_distribution(y_te, y_proba):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # KDE by class
    ax = axes[0]
    for label, color, name in [(0, '#E74C3C', 'Failed'), (1, '#2ECC71', 'Success')]:
        subset = y_proba[y_te == label]
        ax.hist(subset, bins=50, alpha=0.55, color=color,
                label='%s (n=%d)' % (name, len(subset)), density=True)
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(subset)
        xs  = np.linspace(0, 1, 300)
        ax.plot(xs, kde(xs), color=color, lw=2)
    ax.axvline(0.5, color='black', lw=1.5, linestyle='--', label='Threshold=0.50')
    ax.set_xlabel('Predicted Probability of Success')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution by True Class')
    ax.legend()

    # Calibration-style: binned accuracy
    ax2 = axes[1]
    bins = np.linspace(0, 1, 11)
    bin_centers, bin_acc, bin_count = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() > 10:
            bin_centers.append((lo+hi)/2)
            bin_acc.append(y_te[mask].mean())
            bin_count.append(mask.sum())

    ax2.bar(bin_centers, bin_acc, width=0.08, alpha=0.6,
            color='#3498DB', label='Actual success rate')
    ax2.plot(bin_centers, bin_centers, 'k--', lw=1.5, label='Perfect calibration')
    ax2.set_xlabel('Predicted Probability Bin')
    ax2.set_ylabel('Actual Success Rate')
    ax2.set_title('Calibration Plot')
    ax2.legend()
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    path = os.path.join(RES_DIR, 'score_distribution.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved:", path)



def print_summary(y_te, y_pred, y_proba):
    print("\n" + "="*55)
    print("  MODEL EVALUATION SUMMARY")
    print("="*55)
    print("Accuracy  : %.4f" % accuracy_score(y_te, y_pred))
    print("ROC-AUC   : %.4f" % roc_auc_score(y_te, y_proba))
    print("F1 (macro): %.4f" % f1_score(y_te, y_pred, average='macro'))
    print("F1 (Success class): %.4f" % f1_score(y_te, y_pred, pos_label=1))

    top20_t   = np.percentile(y_proba, 80)
    top20_acc = y_te[y_proba >= top20_t].mean()
    baseline  = y_te.mean()
    print("Top-20%% Precision : %.4f  (baseline %.4f)" % (top20_acc, baseline))
    print("Lift               : %.2fx" % (top20_acc / baseline))
    print("\n" + classification_report(y_te, y_pred,
                                       target_names=['Failed', 'Success']))


if __name__ == "__main__":
    print("="*55)
    print("  MODEL EVALUATION")
    print("="*55)

    model, scaler, X_te_s, y_te, y_pred, y_proba, feat_names = load_everything()

    print("\nGenerating plots...")
    plot_confusion_matrix(y_te, y_pred, y_proba)
    plot_roc_curve(y_te, y_proba)
    plot_pr_curve(y_te, y_proba)
    plot_threshold_analysis(y_te, y_proba)
    plot_feature_importance(model, feat_names)
    plot_score_distribution(y_te, y_proba)
    print_summary(y_te, y_pred, y_proba)

    print("\n[DONE] All plots saved to:", RES_DIR)