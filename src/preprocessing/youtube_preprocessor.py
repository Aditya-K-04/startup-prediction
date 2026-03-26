import pandas as pd
import numpy as np
from datetime import datetime

class YouTubePreprocessor:
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
        self.df = None
        
    def load_data(self):
        """Load raw YouTube data"""
        print("Loading YouTube dataset...")
        self.df = pd.read_csv(self.raw_data_path)
        print(f"Loaded {len(self.df)} records")
        return self.df
    
    def handle_missing_values(self):
        """Handle missing values"""
        print("\nHandling missing values...")
        
        # Fill numeric columns with 0
        numeric_cols = ['subscriber_count', 'view_count', 'video_count', 'comment_count']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)
        
        # Fill text columns
        if 'description' in self.df.columns:
            self.df['description'] = self.df['description'].fillna('')
        
        return self.df
    
    def extract_datetime_features(self):
        print("\nExtracting datetime features...")

        if 'published_at' in self.df.columns:
            # Convert to datetime UTC aware
            self.df['published_at'] = pd.to_datetime(self.df['published_at'], errors='coerce', utc=True)

            # Current time with UTC timezone
            now = pd.Timestamp.utcnow()

            # Calculate channel age in days (timezone-aware subtraction)
            self.df['channel_age_days'] = (now - self.df['published_at']).dt.days
            self.df['channel_age_years'] = self.df['channel_age_days'] / 365.25

        return self.df
    
    def calculate_engagement_metrics(self):
        """Calculate engagement metrics"""
        print("\nCalculating engagement metrics...")
        
        # Views per video
        self.df['avg_views_per_video'] = np.where(
            self.df['video_count'] > 0,
            self.df['view_count'] / self.df['video_count'],
            0
        )
        
        # Subscriber engagement rate
        self.df['subscriber_engagement'] = np.where(
            self.df['subscriber_count'] > 0,
            self.df['view_count'] / self.df['subscriber_count'],
            0
        )
        
        # Videos per year
        self.df['videos_per_year'] = np.where(
            self.df['channel_age_years'] > 0,
            self.df['video_count'] / self.df['channel_age_years'],
            0
        )
        
        # Engagement score (composite metric)
        self.df['engagement_score'] = (
            np.log1p(self.df['view_count']) * 0.4 +
            np.log1p(self.df['subscriber_count']) * 0.3 +
            np.log1p(self.df['video_count']) * 0.2 +
            np.log1p(self.df['comment_count']) * 0.1
        )
        
        print("Created engagement metrics")
        
        return self.df
    
    def categorize_channel_size(self):
        """Categorize channels by size"""
        print("\nCategorizing channel sizes...")
        
        # Based on subscriber count
        def categorize_subscribers(count):
            if count < 1000:
                return 'micro'
            elif count < 10000:
                return 'small'
            elif count < 100000:
                return 'medium'
            elif count < 1000000:
                return 'large'
            else:
                return 'mega'
        
        self.df['channel_size'] = self.df['subscriber_count'].apply(categorize_subscribers)
        
        return self.df
    
    def remove_outliers(self):
        """Remove extreme outliers"""
        print("\nRemoving outliers...")
        
        initial_count = len(self.df)
        
        # Remove channels with unrealistic stats
        self.df = self.df[self.df['view_count'] >= 0]
        self.df = self.df[self.df['subscriber_count'] >= 0]
        self.df = self.df[self.df['video_count'] >= 0]
        
        removed = initial_count - len(self.df)
        print(f"Removed {removed} outlier records")
        
        return self.df
    
    def save_processed_data(self, output_path):
        """Save preprocessed data"""
        print(f"\nSaving processed YouTube data to {output_path}...")
        self.df.to_csv(output_path, index=False)
        print("Saved successfully!")
        
    def run_full_pipeline(self, output_path):
        """Run complete preprocessing pipeline"""
        self.load_data()
        self.handle_missing_values()
        self.extract_datetime_features()
        self.calculate_engagement_metrics()
        self.categorize_channel_size()
        self.remove_outliers()
        self.save_processed_data(output_path)
        
        print("\n" + "="*50)
        print("YOUTUBE PREPROCESSING SUMMARY")
        print("="*50)
        print(f"Total records: {len(self.df)}")
        print(f"\nChannel size distribution:")
        print(self.df['channel_size'].value_counts())
        print(f"\nSample records:")
        print(self.df.head())
        
        return self.df


if __name__ == "__main__":
    preprocessor = YouTubePreprocessor('C:/Users/chand/OneDrive/Desktop/7th sem/startup-prediction/data/raw/youtube_channels.csv')
    df = preprocessor.run_full_pipeline('C:/Users/chand/OneDrive/Desktop/7th sem/startup-prediction/data/processed/youtube_processed.csv')
