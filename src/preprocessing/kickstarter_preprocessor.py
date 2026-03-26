import pandas as pd
import numpy as np
from datetime import datetime
import re

class KickstarterPreprocessor:
    def __init__(self, raw_data_path):
        """Initialize with path to raw Kickstarter CSV"""
        self.raw_data_path = raw_data_path
        self.df = None
        
    def load_data(self):
        """Load raw Kickstarter data"""
        print("Loading Kickstarter dataset...")
        self.df = pd.read_csv(self.raw_data_path)
        print(f"Loaded {len(self.df)} records")
        print(f"Columns: {list(self.df.columns)}")
        return self.df
    
    def handle_missing_values(self):
        """Handle missing values in dataset"""
        print("\nHandling missing values...")
        
        # Check missing values
        missing_counts = self.df.isnull().sum()
        print(f"Missing values per column:\n{missing_counts[missing_counts > 0]}")
        
        # Drop rows with critical missing values
        critical_columns = ['name', 'goal', 'state', 'country']
        self.df = self.df.dropna(subset=critical_columns)
        
        # Fill missing category with 'Unknown'
        if 'category' in self.df.columns:
            self.df['category'] = self.df['category'].fillna('Unknown')
        
        print(f"Records after cleaning: {len(self.df)}")
        return self.df
    
    def create_success_label(self):
        """Create binary success label"""
        print("\nCreating success label...")
        
        # Map state to binary success (1) or failure (0)
        success_states = ['successful']
        self.df['success'] = self.df['state'].apply(
            lambda x: 1 if x in success_states else 0
        )
        
        # Keep only successful and failed campaigns (remove live, canceled, suspended)
        self.df = self.df[self.df['state'].isin(['successful', 'failed'])]
        
        success_rate = self.df['success'].mean() * 100
        print(f"Success rate: {success_rate:.2f}%")
        print(f"Successful: {self.df['success'].sum()}, Failed: {(1-self.df['success']).sum()}")
        
        return self.df
    
    def extract_datetime_features(self):
        """Extract features from dates"""
        print("\nExtracting datetime features...")
        
        # Convert to datetime
        if 'launched' in self.df.columns:
            self.df['launched'] = pd.to_datetime(self.df['launched'], errors='coerce')
        if 'deadline' in self.df.columns:
            self.df['deadline'] = pd.to_datetime(self.df['deadline'], errors='coerce')
        
        # Campaign duration in days
        self.df['campaign_duration_days'] = (
            self.df['deadline'] - self.df['launched']
        ).dt.days
        
        # Launch year, month, day of week
        self.df['launch_year'] = self.df['launched'].dt.year
        self.df['launch_month'] = self.df['launched'].dt.month
        self.df['launch_day_of_week'] = self.df['launched'].dt.dayofweek
        
        print("Created: campaign_duration_days, launch_year, launch_month, launch_day_of_week")
        
        return self.df
    
    def calculate_derived_features(self):
        """Calculate derived numerical features"""
        print("\nCalculating derived features...")
        
        # Funding ratio (pledged / goal)
        self.df['funding_ratio'] = self.df['pledged'] / self.df['goal']
        
        # Average pledge per backer
        self.df['avg_pledge_per_backer'] = np.where(
            self.df['backers'] > 0,
            self.df['pledged'] / self.df['backers'],
            0
        )
        
        # Goal category (small, medium, large)
        goal_percentiles = self.df['goal'].quantile([0.33, 0.67])
        self.df['goal_category'] = pd.cut(
            self.df['goal'],
            bins=[0, goal_percentiles[0.33], goal_percentiles[0.67], float('inf')],
            labels=['small', 'medium', 'large']
        )
        
        print("Created: funding_ratio, avg_pledge_per_backer, goal_category")
        
        return self.df
    
    def clean_text_fields(self):
        """Clean and preprocess text fields"""
        print("\nCleaning text fields...")
        
        # Clean name field
        if 'name' in self.df.columns:
            self.df['name_clean'] = self.df['name'].apply(self._clean_text)
            self.df['name_length'] = self.df['name_clean'].apply(len)
            self.df['name_word_count'] = self.df['name_clean'].apply(lambda x: len(x.split()))
        
        # Clean blurb (description) if exists
        if 'blurb' in self.df.columns:
            self.df['blurb_clean'] = self.df['blurb'].fillna('').apply(self._clean_text)
            self.df['blurb_length'] = self.df['blurb_clean'].apply(len)
            self.df['blurb_word_count'] = self.df['blurb_clean'].apply(lambda x: len(x.split()))
        
        print("Created: name_clean, name_length, name_word_count, blurb features")
        
        return self.df
    
    def _clean_text(self, text):
        """Helper function to clean text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def encode_categorical_features(self):
        """Encode categorical variables"""
        print("\nEncoding categorical features...")
        
        # Country encoding
        if 'country' in self.df.columns:
            # Keep top countries, group others as 'Other'
            top_countries = self.df['country'].value_counts().head(10).index
            self.df['country_grouped'] = self.df['country'].apply(
                lambda x: x if x in top_countries else 'Other'
            )
        
        # Category encoding
        if 'category' in self.df.columns:
            # Extract main category if subcategories exist
            self.df['main_category'] = self.df['category'].apply(
                lambda x: x.split('/')[0] if '/' in str(x) else x
            )
        
        print("Created: country_grouped, main_category")
        
        return self.df
    
    def remove_outliers(self):
        """Remove extreme outliers"""
        print("\nRemoving outliers...")
        
        initial_count = len(self.df)
        
        # Remove campaigns with unrealistic goals (e.g., > $10M or < $1)
        self.df = self.df[(self.df['goal'] >= 1) & (self.df['goal'] <= 10000000)]
        
        # Remove campaigns with negative pledged amounts
        self.df = self.df[self.df['pledged'] >= 0]
        
        # Remove campaigns with unrealistic durations
        self.df = self.df[
            (self.df['campaign_duration_days'] > 0) & 
            (self.df['campaign_duration_days'] <= 90)
        ]
        
        removed = initial_count - len(self.df)
        print(f"Removed {removed} outlier records ({removed/initial_count*100:.2f}%)")
        
        return self.df
    
    def select_final_features(self):
        """Select and order final features for modeling"""
        print("\nSelecting final features...")
        
        # Define features to keep
        feature_columns = [
            # Identifiers
            'ID', 'name', 'name_clean',
            
            # Target variable
            'success',
            
            # Numerical features
            'goal', 'pledged', 'backers_count',
            'campaign_duration_days', 'funding_ratio', 'avg_pledge_per_backer',
            'name_length', 'name_word_count',
            
            # Categorical features
            'country', 'country_grouped', 'main_category', 'goal_category',
            
            # Temporal features
            'launch_year', 'launch_month', 'launch_day_of_week',
            'launched', 'deadline'
        ]
        
        # Keep only existing columns
        available_features = [col for col in feature_columns if col in self.df.columns]
        self.df = self.df[available_features]
        
        print(f"Selected {len(available_features)} features")
        
        return self.df
    
    def save_processed_data(self, output_path):
        """Save preprocessed data"""
        print(f"\nSaving preprocessed data to {output_path}...")
        self.df.to_csv(output_path, index=False)
        print("Saved successfully!")
        
    def get_preprocessing_summary(self):
        """Print summary statistics"""
        print("\n" + "="*50)
        print("PREPROCESSING SUMMARY")
        print("="*50)
        print(f"Total records: {len(self.df)}")
        print(f"Total features: {len(self.df.columns)}")
        print(f"\nSuccess rate: {self.df['success'].mean()*100:.2f}%")
        print(f"\nFeature types:")
        print(self.df.dtypes.value_counts())
        print(f"\nSample records:")
        print(self.df.head())
        print(f"\nBasic statistics:")
        print(self.df.describe())
        
    def run_full_pipeline(self, output_path):
        """Run complete preprocessing pipeline"""
        self.load_data()
        self.handle_missing_values()
        self.create_success_label()
        self.extract_datetime_features()
        self.calculate_derived_features()
        self.clean_text_fields()
        self.encode_categorical_features()
        self.remove_outliers()
        self.select_final_features()
        self.save_processed_data(output_path)
        self.get_preprocessing_summary()
        
        return self.df


# Usage example
if __name__ == "__main__":
    preprocessor = KickstarterPreprocessor('C:/Users/chand/OneDrive/Desktop/7th sem/startup-prediction/data/raw/kickstarter_projects.csv')
    df_processed = preprocessor.run_full_pipeline('C:/Users/chand/OneDrive/Desktop/7th sem/startup-prediction/data/processed/kickstarter_processed.csv')
