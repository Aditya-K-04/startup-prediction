# -*- coding: utf-8 -*-
"""
FastAPI backend for Kickstarter Success Predictor
Run: uvicorn main:app --reload --port 8000
"""
import os, warnings, tempfile
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import shap
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import xgboost as xgb

BASE    = r"C:\Users\chand\OneDrive\Desktop\7th sem\startup-prediction"
MDL_DIR = os.path.join(BASE, "models")
DATA    = os.path.join(BASE, "data", "integrated", "final_training_data.csv")

app = FastAPI(title="Kickstarter Success Predictor API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals loaded once at startup ──────────────────────────
MODEL      = None
SCALER     = None
TE         = None
CLEAN_XGB  = None
EXPLAINER  = None
FEAT_NAMES = None
MODEL_STATS= {}


class TargetEncoder:
    def __init__(self, smoothing=10):
        self.smoothing   = smoothing
        self.global_mean = 0.5
        self.encodings   = {}

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


# Category & country mappings for UI dropdowns
MAIN_CATEGORIES = [
    "Art", "Comics", "Crafts", "Dance", "Design", "Fashion",
    "Film & Video", "Food", "Games", "Journalism", "Music",
    "Photography", "Publishing", "Technology", "Theater"
]
COUNTRIES = ["US", "GB", "CA", "AU", "DE", "FR", "IT", "ES", "NL", "SE",
             "MX", "SG", "NZ", "DK", "NO", "AT", "CH", "BE", "IE", "Other"]

CAT_SUCCESS_RATES = {
    "Games": 0.52, "Design": 0.50, "Technology": 0.45, "Comics": 0.55,
    "Music": 0.48, "Film & Video": 0.38, "Art": 0.42, "Publishing": 0.32,
    "Theater": 0.61, "Dance": 0.64, "Fashion": 0.28, "Food": 0.29,
    "Photography": 0.42, "Crafts": 0.31, "Journalism": 0.22
}


@app.on_event("startup")
def load_models():
    global MODEL, SCALER, TE, CLEAN_XGB, EXPLAINER, FEAT_NAMES, MODEL_STATS

    MODEL  = joblib.load(os.path.join(MDL_DIR, "optimized_model.pkl"))
    SCALER = joblib.load(os.path.join(MDL_DIR, "optimized_scaler.pkl"))

    # Load encoder from JSON (avoids pickle class resolution issues)
    te_json_path = os.path.join(MDL_DIR, "target_encoder.json")
    te_pkl_path  = os.path.join(MDL_DIR, "target_encoder.pkl")
    if os.path.exists(te_json_path):
        with open(te_json_path) as f:
            enc_data = json.load(f)
        TE = TargetEncoder(smoothing=enc_data["smoothing"])
        TE.global_mean = enc_data["global_mean"]
        # Convert string keys back to int/float
        TE.encodings = {
            col: {float(k): v for k, v in mapping.items()}
            for col, mapping in enc_data["encodings"].items()
        }
        print("Encoder loaded from JSON OK")
    elif os.path.exists(te_pkl_path):
        print("WARNING: JSON encoder not found, skipping target encoding")
        TE = None

    # SHAP-compatible XGB - save/reload via JSON to fix base_score + use_label_encoder issues
    xgb_model = MODEL.named_estimators_['xgb']
    # Patch out legacy attribute before saving
    if hasattr(xgb_model, 'use_label_encoder'):
        del xgb_model.__dict__['use_label_encoder']
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name
    xgb_model.get_booster().save_model(tmp_path)
    CLEAN_XGB = xgb.XGBClassifier()
    CLEAN_XGB.load_model(tmp_path)
    os.unlink(tmp_path)
    EXPLAINER = shap.TreeExplainer(CLEAN_XGB)

    # Feature names from training data
    df = pd.read_csv(DATA, nrows=5)
    FEAT_NAMES = [c for c in df.columns if c != 'success']

    # Model stats
    MODEL_STATS = {
        "accuracy": 70.5,
        "roc_auc": 0.7737,
        "f1_failed": 0.76,
        "f1_success": 0.62,
        "lift": 1.81,
        "total_training_samples": 331651,
        "features_used": 123,
        "models_in_ensemble": ["XGBoost", "LightGBM", "CatBoost", "RandomForest"],
    }
    print("Models loaded OK")


# ── Request schema ───────────────────────────────────────────
class PredictRequest(BaseModel):
    goal_usd: float           # e.g. 10000
    duration_days: int        # e.g. 30
    main_category: str        # e.g. "Games"
    country: str              # e.g. "US"
    name: Optional[str] = ""  # campaign name
    launch_month: Optional[int] = 6
    launch_weekday: Optional[int] = 1


def build_feature_vector(req: PredictRequest) -> np.ndarray:
    """Build exact 123-column feature vector matching scaler training data."""
    goal_usd  = float(req.goal_usd)
    goal_log  = np.log1p(goal_usd)
    dur_days  = int(req.duration_days)
    dur_log   = np.log1p(dur_days)
    name      = req.name or ""
    cat       = req.main_category
    country   = req.country

    cat_code     = MAIN_CATEGORIES.index(cat) if cat in MAIN_CATEGORIES else 0
    country_idx  = COUNTRIES.index(country)   if country in COUNTRIES   else 0
    cat_rate     = CAT_SUCCESS_RATES.get(cat, 0.4)
    cat_median   = cat_rate * 50000
    cat_med_log  = np.log1p(cat_median)

    row = {}

    # 0-1: goal / duration raw
    row['usd_goal_real'] = goal_usd
    row['duration_days'] = dur_days

    # 2-8: goal features
    row['goal_log']    = goal_log
    row['goal_log_sq'] = goal_log ** 2
    row['goal_tiny']   = int(goal_log < 7.0)
    row['goal_small']  = int(7.0 <= goal_log < 9.2)
    row['goal_medium'] = int(9.2 <= goal_log < 11.5)
    row['goal_large']  = int(11.5 <= goal_log < 13.8)
    row['goal_huge']   = int(goal_log >= 13.8)

    # 9-12: duration features
    row['duration_log']   = dur_log
    row['duration_short'] = int(dur_days <= 15)
    row['duration_ideal'] = int(15 < dur_days <= 35)
    row['duration_long']  = int(dur_days > 35)

    # 13-15: goal relative to category
    row['goal_pct_in_cat']       = float(np.clip(goal_usd / (cat_median + 1), 0, 5))
    row['goal_vs_cat_median']    = np.log1p(goal_usd / (cat_median + 1))
    row['goal_below_cat_median'] = int(goal_usd < cat_median)

    # 16-17: interactions
    row['goal_x_duration'] = goal_log * dur_log
    row['goal_per_day']    = goal_log / (dur_log + 1e-6)

    # 18-24: launch timing
    row['launch_year']      = 2024
    row['launch_month']     = req.launch_month
    row['launch_weekday']   = req.launch_weekday
    row['launch_hour']      = 12
    row['launch_quarter']   = (req.launch_month - 1) // 3 + 1
    row['launch_good_month']= int(req.launch_month in [3,4,5,9,10])
    row['launch_weekend']   = int(req.launch_weekday >= 5)

    # 25-30: name basic features
    row['name_length']      = len(name)
    row['name_word_count']  = len(name.split()) if name else 0
    row['name_has_number']  = int(any(c.isdigit() for c in name))
    row['name_has_exclaim'] = int('!' in name)
    row['name_optimal_len'] = int(15 <= len(name) <= 50)
    row['name_has_colon']   = int(':' in name)

    # 31-38: name keyword features
    for w in ['game','film','music','art','book','app','design','comic']:
        row['name_has_' + w] = int(w in name.lower())

    # 39-40: relaunch
    row['is_relaunch']    = 0
    row['relaunch_count'] = 0

    # 41-44: country / currency flags
    row['is_us']  = int(country == 'US')
    row['is_gb']  = int(country == 'GB')
    row['is_ca']  = int(country == 'CA')
    row['is_usd'] = int(country == 'US')

    # 45-48: label encoded categoricals
    row['main_cat_code'] = cat_code
    row['sub_cat_code']  = cat_code
    row['country_code']  = country_idx
    row['currency_code'] = 0

    # 49-51: category x goal interactions
    row['cat_x_goal']     = cat_code * goal_log
    row['cat_x_duration'] = cat_code * dur_log
    row['goal_per_cat']   = goal_log / (cat_code + 1)

    # 52-66: main category dummies
    for c in ['Art','Comics','Crafts','Dance','Design','Fashion',
              'Film & Video','Food','Games','Journalism','Music',
              'Photography','Publishing','Technology','Theater']:
        row['cat_' + c] = int(cat == c)

    # 67-107: sub-category dummies (all 0 — we don't know sub-cat from UI)
    for s in ['Accessories','Apparel','Apps','Art',"Children's Books",
              'Comic Books','Comics','Country & Folk','Crafts','Design',
              'Documentary','Fashion','Fiction','Film & Video','Food',
              'Games','Hardware','Hip-Hop','Illustration','Indie Rock',
              'Mixed Media','Music','Narrative Film','Nonfiction','Other',
              'Painting','Photography','Pop','Product Design','Public Art',
              'Publishing','Restaurants','Rock','Shorts','Software',
              'Tabletop Games','Technology','Theater','Video Games','Web','Webseries']:
        row['sub_' + s] = 0
    # Set sub-category based on main category as best guess
    sub_map = {
        'Games': 'Tabletop Games', 'Technology': 'Software',
        'Film & Video': 'Narrative Film', 'Music': 'Music',
        'Art': 'Art', 'Design': 'Product Design',
        'Publishing': 'Fiction', 'Fashion': 'Apparel',
        'Food': 'Food', 'Theater': 'Theater',
        'Dance': 'Other', 'Comics': 'Comic Books',
        'Photography': 'Photography', 'Crafts': 'Crafts',
        'Journalism': 'Nonfiction'
    }
    sub_guess = sub_map.get(cat, 'Other')
    if 'sub_' + sub_guess in row:
        row['sub_' + sub_guess] = 1

    # 108-115: reddit + youtube (global constants)
    row['reddit_post_count']   = 35084
    row['reddit_avg_score']    = 15.2
    row['reddit_avg_sentiment']= 0.45
    row['reddit_pos_ratio']    = 0.38
    row['reddit_top_score']    = 1080
    row['yt_subscribers_median']= 347
    row['yt_views_median']     = 17145
    row['yt_channel_count']    = 500

    # 116: GDP (use US average as default)
    row['gdp_growth'] = 2.3

    # 117-122: target encoded columns (all variants)
    row['main_cat_code_te']  = cat_rate
    row['sub_cat_code_te']   = cat_rate
    row['country_code_te']   = 0.4
    row['main_cat_code_te2'] = cat_rate
    row['sub_cat_code_te2']  = cat_rate
    row['country_code_te2']  = 0.4

    # Build DataFrame with exact column order
    COLS = [
        'usd_goal_real','duration_days','goal_log','goal_log_sq','goal_tiny',
        'goal_small','goal_medium','goal_large','goal_huge','duration_log',
        'duration_short','duration_ideal','duration_long','goal_pct_in_cat',
        'goal_vs_cat_median','goal_below_cat_median','goal_x_duration','goal_per_day',
        'launch_year','launch_month','launch_weekday','launch_hour','launch_quarter',
        'launch_good_month','launch_weekend','name_length','name_word_count',
        'name_has_number','name_has_exclaim','name_optimal_len','name_has_colon',
        'name_has_game','name_has_film','name_has_music','name_has_art',
        'name_has_book','name_has_app','name_has_design','name_has_comic',
        'is_relaunch','relaunch_count','is_us','is_gb','is_ca','is_usd',
        'main_cat_code','sub_cat_code','country_code','currency_code',
        'cat_x_goal','cat_x_duration','goal_per_cat',
        'cat_Art','cat_Comics','cat_Crafts','cat_Dance','cat_Design','cat_Fashion',
        'cat_Film & Video','cat_Food','cat_Games','cat_Journalism','cat_Music',
        'cat_Photography','cat_Publishing','cat_Technology','cat_Theater',
        'sub_Accessories','sub_Apparel','sub_Apps','sub_Art',"sub_Children's Books",
        'sub_Comic Books','sub_Comics','sub_Country & Folk','sub_Crafts','sub_Design',
        'sub_Documentary','sub_Fashion','sub_Fiction','sub_Film & Video','sub_Food',
        'sub_Games','sub_Hardware','sub_Hip-Hop','sub_Illustration','sub_Indie Rock',
        'sub_Mixed Media','sub_Music','sub_Narrative Film','sub_Nonfiction','sub_Other',
        'sub_Painting','sub_Photography','sub_Pop','sub_Product Design','sub_Public Art',
        'sub_Publishing','sub_Restaurants','sub_Rock','sub_Shorts','sub_Software',
        'sub_Tabletop Games','sub_Technology','sub_Theater','sub_Video Games',
        'sub_Web','sub_Webseries',
        'reddit_post_count','reddit_avg_score','reddit_avg_sentiment','reddit_pos_ratio',
        'reddit_top_score','yt_subscribers_median','yt_views_median','yt_channel_count',
        'gdp_growth','main_cat_code_te','sub_cat_code_te','country_code_te',
        'main_cat_code_te2','sub_cat_code_te2','country_code_te2'
    ]
    X = pd.DataFrame([{c: row.get(c, 0) for c in COLS}])
    return X


# ── Endpoints ────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Kickstarter Success Predictor API", "version": "1.0.0"}


@app.get("/stats")
def get_stats():
    return MODEL_STATS


@app.get("/categories")
def get_categories():
    return {"categories": MAIN_CATEGORIES, "countries": COUNTRIES}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        X      = build_feature_vector(req)
        X_sc   = SCALER.transform(X)
        proba  = float(CLEAN_XGB.predict_proba(X_sc)[0, 1])
        pred   = int(proba >= 0.5)

        # SHAP explanation
        sv     = EXPLAINER.shap_values(X_sc)
        if isinstance(sv, list):
            sv = sv[1]
        sv = sv[0]

        # Top 8 contributing features
        feat_impact = sorted(
            zip(FEAT_NAMES, sv.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:8]

        shap_explanation = [
            {
                "feature": f,
                "value": float(X[f].values[0]),
                "shap_value": float(s),
                "direction": "positive" if s > 0 else "negative"
            }
            for f, s in feat_impact
        ]

        # Risk level
        if proba >= 0.70:   risk = "Low Risk"
        elif proba >= 0.50: risk = "Moderate Risk"
        elif proba >= 0.35: risk = "High Risk"
        else:               risk = "Very High Risk"

        return {
            "success_probability": round(proba * 100, 1),
            "prediction": "Success" if pred else "Failure",
            "risk_level": risk,
            "shap_explanation": shap_explanation,
            "base_rate": 40.4,
            "vs_baseline": round((proba - 0.404) * 100, 1)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



class RoadmapRequest(BaseModel):
    campaign_name:    str
    category:         str
    goal_usd:         float
    duration_days:    int
    country:          str
    launch_month:     str
    campaign_type:    str
    team_size:        str
    prior_campaigns:  str
    social_followers: str
    prototype_status: str
    has_video:        str
    reward_tiers:     str
    description:      Optional[str] = ""
    success_prob:     float
    risk_level:       str
    top_factors:      str


@app.post("/roadmap")
async def generate_roadmap(req: RoadmapRequest):
    prompt = f"""You are an expert Kickstarter campaign strategist. A creator is planning a campaign:

Campaign Name: "{req.campaign_name}"
Category: {req.category}
Funding Goal: ${req.goal_usd:,.0f}
Duration: {req.duration_days} days
Country: {req.country}
Launch Month: {req.launch_month}
Campaign Type: {req.campaign_type}
Team Size: {req.team_size}
Prior Campaigns: {req.prior_campaigns}
Social Followers: {req.social_followers}
Prototype Status: {req.prototype_status}
Has Video: {req.has_video}
Reward Tiers: {req.reward_tiers}
Description: {req.description or "Not provided"}

ML Model Prediction: {req.success_prob}% success probability ({req.risk_level})
Top influencing factors: {req.top_factors}

Provide a structured campaign improvement roadmap in exactly this format with ## section headers:

## Prediction Summary
Brief 2-sentence analysis of the {req.success_prob}% prediction and what it means for this campaign.

## Critical Improvements
The 3 most impactful changes this creator should make based on their specific inputs and the ML model factors. Be specific and actionable.

## Funding Goal Strategy
Specific advice on whether to raise or lower their ${req.goal_usd:,.0f} goal based on {req.category} category benchmarks and historical data.

## Launch & Timing Strategy
Specific advice on their {req.launch_month} timing, {req.duration_days}-day duration, and optimal day-of-week strategy.

## Community & Marketing Plan
Tailored 3-step pre-launch strategy based on their {req.social_followers} social following and {req.team_size} team size.

## Reward Tier Recommendations
Specific reward structure advice for a {req.category} campaign with {req.reward_tiers} tiers — include suggested price points.

Be direct, specific, and actionable. Reference actual numbers. Write in clear flowing sentences, no bullet points."""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY', '')}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "max_tokens": 1000,
                    "temperature": 0.7,
                    "messages": [
                        {"role": "system", "content": "You are an expert Kickstarter campaign strategist. Give specific, actionable advice."},
                        {"role": "user", "content": prompt}
                    ]
                }
            )
        data = resp.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not text:
            raise ValueError(data.get("error", {}).get("message", "Empty response from Groq"))
        return {"roadmap": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feature-importance")
def feature_importance():
    imp = MODEL.named_estimators_['xgb'].feature_importances_
    fi  = sorted(zip(FEAT_NAMES, imp.tolist()),
                 key=lambda x: x[1], reverse=True)[:20]
    return {
        "features": [{"name": f, "importance": round(v, 5)} for f, v in fi]
    }