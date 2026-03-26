# -*- coding: utf-8 -*-
"""
data_integrator.py  -  Rewritten to preserve all rich features
Outputs a clean, unscaled CSV with 50+ features for train_model.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

BASE  = r"C:\Users\chand\OneDrive\Desktop\7th sem\startup-prediction"
RAW   = os.path.join(BASE, "data", "raw")
PROC  = os.path.join(BASE, "data", "processed")
OUT   = os.path.join(BASE, "data", "integrated", "final_training_data.csv")


# =============================================
# 1. KICKSTARTER  (core features)
# =============================================
def load_kickstarter():
    path = os.path.join(RAW, "kickstarter_projects.csv")
    print("Loading kickstarter_projects.csv ...")
    df = pd.read_csv(path, low_memory=False)
    print("  Raw:", df.shape)

    # --- Target label ---
    df = df[df['state'].isin(['successful', 'failed'])].copy()
    df['success'] = (df['state'] == 'successful').astype(int)

    # --- Goal in USD ---
    if 'usd_goal_real' in df.columns:
        df['goal_usd'] = pd.to_numeric(df['usd_goal_real'], errors='coerce').fillna(0)
    else:
        df['goal_usd'] = pd.to_numeric(df['goal'], errors='coerce').fillna(0)
    df = df[(df['goal_usd'] > 0) & (df['goal_usd'] < 1e8)]

    # --- Dates ---
    df['launched'] = pd.to_datetime(df['launched'], errors='coerce')
    df['deadline'] = pd.to_datetime(df['deadline'], errors='coerce')

    # --- Duration ---
    df['duration_days'] = (df['deadline'] - df['launched']).dt.days.clip(1, 90)

    # --- Goal features ---
    df['goal_log']    = np.log1p(df['goal_usd'])
    df['goal_log_sq'] = df['goal_log'] ** 2
    df['goal_tiny']   = (df['goal_log'] < 7.0).astype(int)
    df['goal_small']  = ((df['goal_log'] >= 7.0)  & (df['goal_log'] < 9.2)).astype(int)
    df['goal_medium'] = ((df['goal_log'] >= 9.2)  & (df['goal_log'] < 11.5)).astype(int)
    df['goal_large']  = ((df['goal_log'] >= 11.5) & (df['goal_log'] < 13.8)).astype(int)
    df['goal_huge']   = (df['goal_log'] >= 13.8).astype(int)

    # --- Duration features ---
    df['duration_log']   = np.log1p(df['duration_days'])
    df['duration_short'] = (df['duration_days'] <= 15).astype(int)
    df['duration_ideal'] = ((df['duration_days'] > 15) & (df['duration_days'] <= 35)).astype(int)
    df['duration_long']  = (df['duration_days'] > 35).astype(int)

    # --- Goal relative to category ---
    cat_median = df.groupby('main_category')['goal_usd'].transform('median')
    df['goal_pct_in_cat']      = df.groupby('main_category')['goal_usd'].rank(pct=True)
    df['goal_vs_cat_median']   = np.log1p(df['goal_usd'] / (cat_median + 1))
    df['goal_below_cat_median']= (df['goal_usd'] < cat_median).astype(int)

    # --- Goal x Duration interactions ---
    df['goal_x_duration'] = df['goal_log'] * df['duration_log']
    df['goal_per_day']    = df['goal_log'] / (df['duration_log'] + 1e-6)

    # --- Launch timing ---
    df['launch_year']      = df['launched'].dt.year.fillna(2015).astype(int)
    df['launch_month']     = df['launched'].dt.month.fillna(6).astype(int)
    df['launch_weekday']   = df['launched'].dt.weekday.fillna(2).astype(int)
    df['launch_hour']      = df['launched'].dt.hour.fillna(12).astype(int)
    df['launch_quarter']   = df['launched'].dt.quarter.fillna(2).astype(int)
    df['launch_good_month']= df['launched'].dt.month.isin([3,4,5,9,10]).fillna(False).astype(int)
    df['launch_weekend']   = (df['launched'].dt.weekday >= 5).fillna(False).astype(int)

    # --- Name features ---
    names = df['name'].fillna('').astype(str)
    df['name_length']      = names.str.len()
    df['name_word_count']  = names.str.split().str.len()
    df['name_has_number']  = names.str.contains(r'\d', regex=True).astype(int)
    df['name_has_exclaim'] = names.str.contains('!').astype(int)
    df['name_optimal_len'] = ((names.str.len() >= 15) & (names.str.len() <= 50)).astype(int)
    df['name_has_colon']   = names.str.contains(':').astype(int)
    for w in ['game', 'film', 'music', 'art', 'book', 'app', 'design', 'comic']:
        df['name_has_' + w] = names.str.lower().str.contains(w).astype(int)

    # --- Relaunch flag ---
    name_counts = names.map(names.value_counts())
    df['is_relaunch']    = (name_counts > 1).astype(int)
    df['relaunch_count'] = (name_counts - 1).clip(0, 5)

    # --- Country flags ---
    df['is_us'] = (df['country'] == 'US').astype(int)
    df['is_gb'] = (df['country'] == 'GB').astype(int)
    df['is_ca'] = (df['country'] == 'CA').astype(int)
    df['is_usd']= (df['currency'] == 'USD').astype(int)

    # --- Label encode categoricals (raw codes, NOT scaled) ---
    df['main_cat_code'] = df['main_category'].astype('category').cat.codes
    df['sub_cat_code']  = df['category'].astype('category').cat.codes
    df['country_code']  = df['country'].astype('category').cat.codes
    df['currency_code'] = df['currency'].astype('category').cat.codes

    # --- Category x Goal interactions ---
    df['cat_x_goal']     = df['main_cat_code'] * df['goal_log']
    df['cat_x_duration'] = df['main_cat_code'] * df['duration_log']
    df['goal_per_cat']   = df['goal_log'] / (df['main_cat_code'].replace(0, 1))

    # --- Main category dummies ---
    cat_dummies = pd.get_dummies(df['main_category'], prefix='cat')
    df = pd.concat([df, cat_dummies], axis=1)

    # --- Top 40 sub-category dummies ---
    top40 = df['category'].value_counts().head(40).index
    df['_sub_top'] = df['category'].where(df['category'].isin(top40), other='Other')
    sub_dummies = pd.get_dummies(df['_sub_top'], prefix='sub')
    df = pd.concat([df, sub_dummies], axis=1)
    df.drop(columns=['_sub_top'], inplace=True)

    print("  After engineering:", df.shape,
          "| Success: %.1f%%" % (df['success'].mean()*100))
    return df


# =============================================
# 2. REDDIT  (aggregate sentiment stats)
# =============================================
def load_reddit():
    path = os.path.join(RAW, "reddit_posts.csv")
    print("Loading reddit_posts.csv ...")
    try:
        rd = pd.read_csv(path, low_memory=False)
        rd.columns = rd.columns.str.strip().str.lower()
        stats = {
            'reddit_post_count'      : len(rd),
            'reddit_avg_score'       : rd['score'].mean()           if 'score'           in rd.columns else 0,
            'reddit_avg_sentiment'   : rd['sentiment_score'].mean() if 'sentiment_score' in rd.columns else 0,
            'reddit_pos_ratio'       : (rd['sentiment']=='POSITIVE').mean() if 'sentiment' in rd.columns else 0,
            'reddit_top_score'       : rd['score'].max()            if 'score'           in rd.columns else 0,
        }
        print("  Posts:", len(rd), "| Avg score: %.2f" % stats['reddit_avg_score'])
        return stats
    except Exception as e:
        print("  Reddit failed:", e)
        return {}


# =============================================
# 3. YOUTUBE  (aggregate channel stats)
# =============================================
def load_youtube():
    path = os.path.join(RAW, "youtube_channels.csv")
    print("Loading youtube_channels.csv ...")
    try:
        yt = pd.read_csv(path, low_memory=False)
        yt.columns = yt.columns.str.strip().str.lower()
        stats = {
            'yt_subscribers_median': yt['subscriber_count'].median() if 'subscriber_count' in yt.columns else 0,
            'yt_views_median'      : yt['view_count'].median()       if 'view_count'       in yt.columns else 0,
            'yt_channel_count'     : len(yt),
        }
        print("  Channels:", len(yt))
        return stats
    except Exception as e:
        print("  YouTube failed:", e)
        return {}


# =============================================
# 4. GDP  (join by country + launch year)
# =============================================
def load_gdp():
    path = os.path.join(RAW, "worldbank_gdp_growth.csv")
    print("Loading worldbank_gdp_growth.csv ...")
    try:
        raw = pd.read_csv(path, skiprows=4, low_memory=False)
        raw.columns = raw.columns.str.strip()
        year_cols = [c for c in raw.columns if c.isdigit()]
        gdp_long  = raw.melt(
            id_vars=['Country Code'], value_vars=year_cols,
            var_name='year', value_name='gdp_growth'
        )
        gdp_long['year']       = gdp_long['year'].astype(int)
        gdp_long['gdp_growth'] = pd.to_numeric(gdp_long['gdp_growth'], errors='coerce')
        gdp_long = gdp_long.dropna(subset=['gdp_growth'])
        print("  GDP rows:", len(gdp_long))
        return gdp_long.set_index(['Country Code', 'year'])['gdp_growth']
    except Exception as e:
        print("  GDP failed:", e)
        return None


# =============================================
# 5. TARGET ENCODING  (no leakage - done here
#    on full dataset; train_model.py will also
#    do it leak-free on train split only)
# =============================================
def target_encode(df, cols, y, smoothing=10):
    global_mean = y.mean()
    for col in cols:
        if col not in df.columns:
            continue
        tmp   = pd.DataFrame({'x': df[col], 'y': y.values})
        stats = tmp.groupby('x')['y'].agg(['mean', 'count'])
        enc   = ((stats['mean'] * stats['count'] + global_mean * smoothing)
                 / (stats['count'] + smoothing))
        df[col + '_te'] = df[col].map(enc).fillna(global_mean)
    return df


# =============================================
# MAIN
# =============================================
def run():
    print("=" * 55)
    print("  DATA INTEGRATION PIPELINE  (rewritten)")
    print("=" * 55)

    # Load and engineer all sources
    df          = load_kickstarter()
    reddit_stats= load_reddit()
    yt_stats    = load_youtube()
    gdp_index   = load_gdp()

    # Add scalar stats from Reddit + YouTube as columns
    for k, v in {**reddit_stats, **yt_stats}.items():
        df[k] = v

    # GDP join: country ISO2 -> ISO3 -> lookup by year
    if gdp_index is not None and 'launch_year' in df.columns:
        iso2_to_iso3 = {
            'US':'USA','GB':'GBR','CA':'CAN','AU':'AUS','DE':'DEU',
            'FR':'FRA','IT':'ITA','ES':'ESP','NL':'NLD','SE':'SWE',
            'MX':'MEX','SG':'SGP','HK':'HKG','NZ':'NZL','DK':'DNK',
            'NO':'NOR','AT':'AUT','CH':'CHE','BE':'BEL','IE':'IRL',
            'JP':'JPN','KR':'KOR','BR':'BRA','IN':'IND','CN':'CHN'
        }
        wb3 = df['country'].map(iso2_to_iso3).fillna('USA')
        df['gdp_growth'] = [
            gdp_index.get((c, y), np.nan)
            for c, y in zip(wb3, df['launch_year'])
        ]
        df['gdp_growth'] = df['gdp_growth'].fillna(df['gdp_growth'].median())
        print("  GDP joined.")

    # Target encode high-cardinality categoricals
    y = df['success']
    df = target_encode(df, ['main_cat_code', 'sub_cat_code', 'country_code'], y)

    # Drop raw/leaky/string columns before saving
    drop_cols = [
        'ID', 'name', 'category', 'main_category', 'currency',
        'launched', 'deadline', 'state', 'pledged', 'backers',
        'usd pledged', 'usd_pledged_real', 'goal', 'goal_usd',
        'country',   # string - already encoded as country_code + is_us/gb/ca
        # Post-launch leaky signals
        'funding_ratio', 'avg_pledge_per_backer',
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Final safety: drop any remaining object/string columns
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    if obj_cols:
        print("  Dropping remaining string cols:", obj_cols)
        df.drop(columns=obj_cols, inplace=True)

    # Save
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print("\n" + "=" * 55)
    print("  INTEGRATION COMPLETE")
    print("=" * 55)
    print("Shape         :", df.shape)
    print("Features      :", df.shape[1] - 1)
    print("Success rate  : %.1f%%" % (df['success'].mean()*100))
    print("Saved to      :", OUT)
    print("\nTop 10 columns:", df.columns[:10].tolist())
    return df


if __name__ == "__main__":
    run()