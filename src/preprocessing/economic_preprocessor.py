import pandas as pd
import numpy as np
import os

class EconomicPreprocessor:
    def __init__(self, raw_path):
        self.raw_path = raw_path
        self.df = None
        
    def load_data(self):
        """Load ANY World Bank CSV - handles all formats"""
        print("Loading World Bank GDP data...")
        
        # Try different parsing strategies
        try:
            self.df = pd.read_csv(self.raw_path, skiprows=4)  # Standard World Bank
        except:
            try:
                self.df = pd.read_csv(self.raw_path)  # Simple CSV
            except:
                # Manual simple format (your copy-paste CSV)
                self.df = pd.read_csv(self.raw_path)
        
        print(f"Raw shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns[:10])}...")
        return self
    
    def standardize_countries(self):
        """Standardize country names/codes to match Kickstarter"""
        print("Standardizing countries...")
        
        # Find country columns (first 2-3 columns usually)
        country_cols = self.df.columns[:3]
        self.df.columns = ['country_name', 'country_code'] + list(self.df.columns[2:])
        
        # Common mappings
        mapping = {
            'United States': 'US', 'United Kingdom': 'GB', 'Canada': 'CA',
            'Australia': 'AU', 'Germany': 'DE', 'France': 'FR',
            'Netherlands': 'NL', 'Italy': 'IT', 'Spain': 'ES', 'Japan': 'JP'
        }
        
        self.df['country_code'] = self.df['country_code'].map(mapping).fillna(self.df['country_code'])
        self.df['country_name'] = self.df['country_name'].replace(mapping)
        
        return self
    
    def extract_recent_gdp(self, years=[2020,2021,2022,2023]):
        """Extract recent GDP growth data"""
        print("Extracting recent GDP data...")
        
        # Find year columns
        year_cols = [col for col in self.df.columns if any(str(y) in str(col) for y in years)]
        
        # Melt to long format
        self.df_long = self.df.melt(
            id_vars=['country_name', 'country_code'],
            value_vars=year_cols[:10],  # Top 10 year columns
            var_name='year',
            value_name='gdp_growth'
        )
        
        # Clean numeric
        self.df_long['gdp_growth'] = pd.to_numeric(self.df_long['gdp_growth'], errors='coerce')
        self.df_long = self.df_long.dropna(subset=['gdp_growth'])
        
        print(f"Long format: {len(self.df_long)} records")
        return self
    
    def create_country_features(self):
        """Create final features per country"""
        print("Creating country-level features...")
        
        # Latest GDP growth (most recent year per country)
        latest = self.df_long.loc[self.df_long.groupby('country_code')['year'].idxmax()]
        
        # 3-year average
        avg_gdp = self.df_long.groupby('country_code')['gdp_growth'].agg(['mean', 'std']).round(2)
        avg_gdp.columns = ['avg_gdp_growth', 'gdp_volatility']
        
        # Combine
        self.features = latest.set_index('country_code')[['gdp_growth']].join(avg_gdp)
        self.features.columns = ['latest_gdp_growth', 'avg_gdp_growth', 'gdp_volatility']
        self.features = self.features.fillna(0)
        
        print(f"Features: {self.features.shape}")
        print(self.features.head())
        return self
    
    def save_processed(self, output_path):
        """Save features"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.features.to_csv(output_path)
        print(f"Saved: {output_path}")
        return self.features
    
    def run_full_pipeline(self, output_path):
        """Complete pipeline"""
        self.load_data()
        self.standardize_countries()
        self.extract_recent_gdp()
        self.create_country_features()
        return self.save_processed(output_path)

if __name__ == "__main__":
    preprocessor = EconomicPreprocessor("C:/Users/chand/OneDrive/Desktop/7th sem/startup-prediction/data/raw/worldbank_gdp_growth.csv")
    preprocessor.run_full_pipeline("C:/Users/chand/OneDrive/Desktop/7th sem/startup-prediction/data/processed/economic_indicators.csv")
    print("\nEconomic features ready for integration!")



