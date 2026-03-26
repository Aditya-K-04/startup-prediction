import pandas as pd
import numpy as np

class RedditPreprocessor:
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
        self.df = None

    def load_data(self):
        print("Loading Reddit dataset...")
        self.df = pd.read_csv(self.raw_data_path)
        print(f"Loaded {len(self.df)} rows")
        print("Columns:", list(self.df.columns))
        return self.df

    def select_and_rename_columns(self):
        """
        Adapt this mapping based on the Kaggle file you use.
        For reddit-dataset-with-sentiment-analysis user_posts.csv:
        - Title, Subreddit, Score, URL, Sentiment, Sentiment_Score, etc.
        """
        print("\nSelecting and renaming columns...")

        # Try flexible mapping so it doesn't crash
        col_map_options = {
            "title": ["Title", "title", "Post_Title"],
            "subreddit": ["Subreddit", "subreddit"],
            "score": ["Score", "score"],
            "url": ["URL", "url", "Link"],
            "sentiment_label": ["Sentiment", "sentiment"],
            "sentiment_score": ["Sentiment_Score", "sentiment_score", "SentimentScore"],
            "text": ["Text", "Body", "Post_Text", "selftext", "Comment_Text"]
        }

        chosen_cols = {}
        for target, candidates in col_map_options.items():
            for c in candidates:
                if c in self.df.columns:
                    chosen_cols[target] = c
                    break

        print("Mapped columns:", chosen_cols)

        # Keep only what we mapped
        self.df = self.df[list(chosen_cols.values())].rename(columns={
            v: k for k, v in chosen_cols.items()
        })

        return self.df

    def handle_missing(self):
        print("\nHandling missing values...")
        if "text" in self.df.columns:
            self.df["text"] = self.df["text"].fillna("")
        if "sentiment_score" in self.df.columns:
            self.df["sentiment_score"] = self.df["sentiment_score"].fillna(0.0)
        if "sentiment_label" in self.df.columns:
            self.df["sentiment_label"] = self.df["sentiment_label"].fillna("neutral")
        return self.df

    def basic_features(self):
        print("\nCreating basic Reddit features...")

        if "text" in self.df.columns:
            self.df["text_length"] = self.df["text"].astype(str).apply(len)
            self.df["word_count"] = self.df["text"].astype(str).apply(
                lambda x: len(x.split())
            )

        if "score" in self.df.columns:
            self.df["is_popular"] = (self.df["score"] >= self.df["score"].median()).astype(int)

        return self.df

    def save(self, out_path):
        print(f"\nSaving processed Reddit data to {out_path}...")
        self.df.to_csv(out_path, index=False)
        print("Saved.")

    def summary(self):
        print("\n========== REDDIT PREPROCESS SUMMARY ==========")
        print("Rows:", len(self.df))
        print("Columns:", len(self.df.columns))
        if "sentiment_label" in self.df.columns:
            print("\nSentiment distribution:")
            print(self.df["sentiment_label"].value_counts())
        if "subreddit" in self.df.columns:
            print("\nTop subreddits:")
            print(self.df["subreddit"].value_counts().head())

    def run_full_pipeline(self, out_path):
        self.load_data()
        self.select_and_rename_columns()
        self.handle_missing()
        self.basic_features()
        self.save(out_path)
        self.summary()
        return self.df


if __name__ == "__main__":
    pre = RedditPreprocessor("C:/Users/chand/OneDrive/Desktop/7th sem/startup-prediction/data/raw/reddit_posts.csv")
    pre.run_full_pipeline("C:/Users/chand/OneDrive/Desktop/7th sem/startup-prediction/data/processed/reddit_processed.csv")
