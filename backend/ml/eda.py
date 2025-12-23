import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 6)

def load_raw_data():
    """Load raw weather data"""
    print("📂 Loading raw data...")
    df = pd.read_csv("data/raw/tokyo_weather_raw.csv", index_col="date", parse_dates=True)
    print(f"✅ Loaded {len(df)} records")
    return df

def check_missing_values(df):
    """Check for missing values"""
    print("\n🔍 Checking for missing values...")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✅ No missing values!")
    else:
        print(f"⚠️  Missing values:\n{missing}")
    return missing

def fill_missing_values(df):
    """Fill missing values using forward fill"""
    print("\n🔧 Filling missing values...")
    df = df.fillna(method='ffill').fillna(method='bfill')
    print("✅ Missing values filled")
    return df

def plot_time_series(df):
    """Visualize temperature trends"""
    print("\n📊 Creating time series plots...")
    
    os.makedirs("figures", exist_ok=True)
    
    # Plot all variables
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    axes[0, 0].plot(df.index, df['temp_mean'], color='#3b82f6', linewidth=1)
    axes[0, 0].set_title('Mean Temperature Over Time')
    axes[0, 0].set_ylabel('Temperature (°C)')
    
    axes[0, 1].plot(df.index, df['temp_max'], color='#ef4444', alpha=0.7, label='Max')
    axes[0, 1].plot(df.index, df['temp_min'], color='#3b82f6', alpha=0.7, label='Min')
    axes[0, 1].set_title('Daily Temperature Range')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].legend()
    
    axes[1, 0].plot(df.index, df['precipitation'], color='#10b981', linewidth=0.8)
    axes[1, 0].set_title('Daily Precipitation')
    axes[1, 0].set_ylabel('Precipitation (mm)')
    
    axes[1, 1].plot(df.index, df['wind_speed_max'], color='#f59e0b', linewidth=0.8)
    axes[1, 1].set_title('Wind Speed (Max)')
    axes[1, 1].set_ylabel('Wind Speed (m/s)')
    
    axes[2, 0].plot(df.index, df['precipitation'], color='#8b5cf6', linewidth=0.8)
    axes[2, 0].set_title('Precipitation')
    axes[2, 0].set_ylabel('Precipitation (mm)')
    
    axes[2, 1].remove()
    
    plt.tight_layout()
    plt.savefig('figures/01_time_series.png', dpi=100, bbox_inches='tight')
    print("✅ Saved: figures/01_time_series.png")
    plt.close()

def plot_seasonal_decomposition(df):
    """Plot seasonal patterns"""
    print("\n📈 Creating seasonal decomposition plot...")
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    decomposition = seasonal_decompose(df['temp_mean'], model='additive', period=365)
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 10))
    
    decomposition.observed.plot(ax=axes[0], color='#3b82f6')
    axes[0].set_title('Observed')
    axes[0].set_ylabel('Temperature (°C)')
    
    decomposition.trend.plot(ax=axes[1], color='#ef4444')
    axes[1].set_title('Trend')
    axes[1].set_ylabel('Temperature (°C)')
    
    decomposition.seasonal.plot(ax=axes[2], color='#10b981')
    axes[2].set_title('Seasonal (365-day cycle)')
    axes[2].set_ylabel('Temperature (°C)')
    
    decomposition.resid.plot(ax=axes[3], color='#f59e0b')
    axes[3].set_title('Residual')
    axes[3].set_ylabel('Temperature (°C)')
    
    plt.tight_layout()
    plt.savefig('figures/02_seasonal_decomposition.png', dpi=100, bbox_inches='tight')
    print("✅ Saved: figures/02_seasonal_decomposition.png")
    plt.close()

def plot_acf_pacf(df):
    """Plot ACF and PACF for stationarity analysis"""
    print("\n📊 Creating ACF/PACF plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    plot_acf(df['temp_mean'].dropna(), lags=40, ax=axes[0])
    axes[0].set_title('Autocorrelation Function (ACF)')
    
    plot_pacf(df['temp_mean'].dropna(), lags=40, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation Function (PACF)')
    
    plt.tight_layout()
    plt.savefig('figures/03_acf_pacf.png', dpi=100, bbox_inches='tight')
    print("✅ Saved: figures/03_acf_pacf.png")
    plt.close()

def stationarity_test(df):
    """Perform Augmented Dickey-Fuller test"""
    print("\n🧪 Performing Augmented Dickey-Fuller test...")
    
    result = adfuller(df['temp_mean'].dropna())
    
    print(f"ADF Statistic: {result[0]:.6f}")
    print(f"P-value: {result[1]:.6f}")
    print(f"Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.3f}")
    
    if result[1] <= 0.05:
        print("✅ Data is stationary (p-value <= 0.05)")
    else:
        print("⚠️  Data is NOT stationary, differencing may be needed")
    
    return result

def print_statistics(df):
    """Print summary statistics"""
    print("\n📊 Summary Statistics:")
    print(df.describe())

def process_and_save(df):
    """Process and save cleaned data"""
    print("\n💾 Saving processed data...")
    
    os.makedirs("data/processed", exist_ok=True)
    
    processed_path = "data/processed/tokyo_weather_processed.csv"
    df.to_csv(processed_path)
    print(f"✅ Saved to: {processed_path}")
    
    return df

def main():
    """Run complete EDA pipeline"""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS - TOKYO WEATHER")
    print("=" * 60)
    
    # Load data
    df = load_raw_data()
    
    # Check missing values
    check_missing_values(df)
    
    # Fill missing values
    df = fill_missing_values(df)
    
    # Print statistics
    print_statistics(df)
    
    # Visualizations
    plot_time_series(df)
    plot_seasonal_decomposition(df)
    plot_acf_pacf(df)
    
    # Stationarity test
    stationarity_test(df)
    
    # Save processed data
    process_and_save(df)
    
    print("\n" + "=" * 60)
    print("✅ EDA Complete! Check 'figures/' folder for visualizations")
    print("=" * 60)

if __name__ == "__main__":
    main()