import pandas as pd
import requests
import zipfile
import io
import os

def download_worldbank_gdp():
    """Download World Bank GDP Growth data"""
    print("Downloading World Bank GDP Growth data...")
    
    # Direct CSV URL (most recent data)
    url = "https://api.worldbank.org/v2/en/indicator/NY.GDP.MKTP.KD.ZG?downloadformat=csv"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Extract ZIP and get main data file
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Find the main data file (usually API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_xxxx.csv)
            data_files = [f for f in z.namelist() if f.endswith('.csv') and 'Metadata' not in f]
            if data_files:
                data_file = data_files[0]
                df = pd.read_csv(z.open(data_file), skiprows=4)
                print(f"Downloaded: {data_file}")
            else:
                print("No data file found")
                return None
        
        # Save raw data
        os.makedirs("../../data/raw", exist_ok=True)
        output_path = "../../data/raw/worldbank_gdp_growth.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        print(f"Shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Download failed: {e}")
        return None

if __name__ == "__main__":
    df = download_worldbank_gdp()
