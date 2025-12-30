import pandas as pd
import os
from datetime import datetime

def load_california_data(csv_path="data/raw/california_load_raw.csv"):
    """
    Load California electricity + weather data from Kaggle CSV (Database.csv)
    
    Dataset contains hourly observations with:
    - Time: timestamp
    - Season: 1-4 (Winter, Spring, Summer, Autumn)
    - Day_of_the_week: 0-6 (Monday-Sunday)
    - DHI, DNI, GHI: Solar radiation (W/m2)
    - Wind_speed: m/s
    - Humidity: %
    - Temperature: degrees (°C)
    - PV_production: Solar production (MW)
    - Wind_production: Wind production (MW)
    - Electric_demand: Electricity demand (MW) - TARGET
    """
    print("=" * 80)
    print("📂 CALIFORNIA GRID LOAD FORECASTING - DATA LOADER")
    print("=" * 80)
    
    try:
        print(f"\n📂 Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"✅ Data loaded successfully!\n")
        
        # Display basic info
        print(f"📊 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"⏰ Hourly observations: {len(df):,}")
        
        # Convert Time to datetime
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            date_min = df['Time'].min()
            date_max = df['Time'].max()
            print(f"📅 Date range: {date_min} to {date_max}")
            print(f"📈 Duration: {(date_max - date_min).days} days (~{(date_max - date_min).days / 365:.1f} years)")
        
        # Display all columns
        print(f"\n📋 Available Columns ({df.shape[1]}):")
        for i, col in enumerate(df.columns, 1):
            dtype = df[col].dtype
            non_null = df[col].notna().sum()
            print(f"   {i:2d}. {col:30s} ({dtype}, {non_null:,} non-null values)")
        
        # Check for missing values
        print(f"\n🔍 Missing Values:")
        missing_count = df.isnull().sum()
        if missing_count.sum() == 0:
            print("   ✅ No missing values detected!")
        else:
            missing_pct = (missing_count / len(df)) * 100
            for col in missing_count[missing_count > 0].index:
                print(f"   ⚠️  {col}: {missing_count[col]:,} ({missing_pct[col]:.2f}%)")
        
        # Statistics for key columns
        print(f"\n📊 Data Summary (Key Variables):")
        key_cols = ['Electric_demand', 'Temperature', 'Humidity', 'Wind_speed']
        available_cols = [col for col in key_cols if col in df.columns]
        if available_cols:
            print(df[available_cols].describe().round(2))
        
        # Check target variable
        if 'Electric_demand' in df.columns:
            demand = df['Electric_demand']
            print(f"\n🎯 TARGET VARIABLE - Electricity Demand (MW):")
            print(f"   Min:  {demand.min():,.2f} MW")
            print(f"   Max:  {demand.max():,.2f} MW")
            print(f"   Mean: {demand.mean():,.2f} MW")
            print(f"   Std:  {demand.std():,.2f} MW")
        
        # Create processed directory
        os.makedirs("data/processed", exist_ok=True)
        
        # Save processed version
        processed_path = "data/processed/california_data_processed.csv"
        df.to_csv(processed_path, index=False)
        print(f"\n💾 Processed data saved to: {processed_path}")
        
        # Summary
        print("\n" + "=" * 80)
        print("✅ DATA LOADING COMPLETE")
        print("=" * 80)
        print(f"\n📌 Next Steps:")
        print(f"   1. Run EDA analysis:  python eda.py")
        print(f"   2. Build SARIMA model: python sarima_model.py")
        print(f"   3. Deploy to production")
        
        return df
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: File not found!")
        print(f"   Expected path: {csv_path}")
        print(f"\n📥 Please ensure your Kaggle CSV is in the correct location:")
        print(f"   1. Download 'Database.csv' from Kaggle")
        print(f"   2. Copy to: {csv_path}")
        print(f"\n💡 Command to copy from Downloads (macOS/Linux):")
        print(f"   mkdir -p data/raw")
        print(f"   cp ~/Downloads/Database.csv {csv_path}")
        print(f"\n💡 Command (Windows):")
        print(f"   mkdir data\\raw")
        print(f"   copy %USERPROFILE%\\Downloads\\Database.csv {csv_path}")
        return None
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return None


if __name__ == "__main__":
    # Load the data
    df = load_california_data()
    
    if df is not None:
        print("\n🚀 Data is ready for analysis!")