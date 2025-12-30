import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import os

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 6)

def load_data():
    """Load processed California data"""
    print("\n" + "=" * 80)
    print("📊 EXPLORATORY DATA ANALYSIS - CALIFORNIA GRID LOAD FORECASTING")
    print("=" * 80)
    
    df = pd.read_csv('data/processed/california_data_processed.csv')
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    
    print(f"\n✅ Data loaded: {len(df):,} observations")
    return df

def create_figures_dir():
    """Create figures directory"""
    os.makedirs('figures', exist_ok=True)

def plot_time_series(df):
    """Plot electricity demand over time"""
    print("\n📈 Creating time series plot...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Electricity demand
    axes[0].plot(df['Time'], df['Electric_demand'], color='#1f77b4', linewidth=0.8)
    axes[0].set_title('Electricity Demand Over Time', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Demand (MW)')
    axes[0].grid(True, alpha=0.3)
    
    # Temperature
    axes[1].plot(df['Time'], df['Temperature'], color='#ff7f0e', linewidth=0.8)
    axes[1].set_title('Temperature Over Time', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].grid(True, alpha=0.3)
    
    # Solar + Wind Production
    axes[2].plot(df['Time'], df['PV_production'], label='Solar', color='#2ca02c', linewidth=0.8)
    axes[2].plot(df['Time'], df['Wind_production'], label='Wind', color='#d62728', linewidth=0.8)
    axes[2].set_title('Solar & Wind Production Over Time', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Production (MW)')
    axes[2].set_xlabel('Time')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/01_time_series.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: figures/01_time_series.png")
    plt.close()

def plot_seasonal_decomposition(df):
    """Decompose time series into components"""
    print("\n📉 Creating seasonal decomposition plot...")
    
    # Use daily averages for cleaner decomposition
    df_daily = df.set_index('Time')['Electric_demand'].resample('D').mean()
    
    # Perform decomposition
    decomposition = seasonal_decompose(df_daily, model='additive', period=365)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    axes[0].plot(decomposition.observed, color='#1f77b4')
    axes[0].set_title('Observed (Daily Average)', fontweight='bold')
    axes[0].set_ylabel('Demand (MW)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(decomposition.trend, color='#ff7f0e')
    axes[1].set_title('Trend', fontweight='bold')
    axes[1].set_ylabel('Demand (MW)')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(decomposition.seasonal, color='#2ca02c')
    axes[2].set_title('Seasonality (365-day cycle)', fontweight='bold')
    axes[2].set_ylabel('Demand (MW)')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(decomposition.resid, color='#d62728')
    axes[3].set_title('Residuals', fontweight='bold')
    axes[3].set_ylabel('Demand (MW)')
    axes[3].set_xlabel('Date')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/02_seasonal_decomposition.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: figures/02_seasonal_decomposition.png")
    plt.close()

def plot_acf_pacf(df):
    """Plot ACF and PACF for SARIMA parameter selection"""
    print("\n🔍 Creating ACF/PACF plots...")
    
    # Use daily data for cleaner plots
    daily_demand = df.set_index('Time')['Electric_demand'].resample('D').mean()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # ACF
    plot_acf(daily_demand, lags=60, ax=axes[0], title='Autocorrelation Function (ACF)')
    axes[0].set_ylabel('ACF')
    axes[0].grid(True, alpha=0.3)
    
    # PACF
    plot_pacf(daily_demand, lags=60, ax=axes[1], title='Partial Autocorrelation Function (PACF)')
    axes[1].set_ylabel('PACF')
    axes[1].set_xlabel('Lags')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/03_acf_pacf.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: figures/03_acf_pacf.png")
    plt.close()

def adf_test(df):
    """Augmented Dickey-Fuller test for stationarity"""
    print("\n🧪 Augmented Dickey-Fuller Test:")
    print("-" * 60)
    
    # Use daily data
    daily_demand = df.set_index('Time')['Electric_demand'].resample('D').mean()
    
    result = adfuller(daily_demand, autolag='AIC')
    
    print(f"ADF Statistic: {result[0]:.6f}")
    print(f"P-value: {result[1]:.6f}")
    print(f"Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.3f}")
    
    if result[1] <= 0.05:
        print("\n✅ Data is STATIONARY (p-value ≤ 0.05)")
        print("   → Can proceed with ARIMA/SARIMA directly")
    else:
        print("\n⚠️  Data is NON-STATIONARY (p-value > 0.05)")
        print("   → Differencing may be needed (d=1 or d=2)")

def correlation_analysis(df):
    """Analyze correlation with demand"""
    print("\n📊 Correlation with Electricity Demand:")
    print("-" * 60)
    
    corr_cols = ['Temperature', 'Humidity', 'Wind_speed', 'PV_production', 'Wind_production']
    correlations = df[corr_cols + ['Electric_demand']].corr()['Electric_demand'].drop('Electric_demand')
    
    correlations_sorted = correlations.sort_values(ascending=False)
    
    for col, corr in correlations_sorted.items():
        print(f"{col:20s}: {corr:7.4f}")
    
    # Plot correlations
    fig, ax = plt.subplots(figsize=(10, 6))
    correlations_sorted.plot(kind='barh', ax=ax, color=['green' if x > 0 else 'red' for x in correlations_sorted])
    ax.set_title('Feature Correlation with Electricity Demand', fontweight='bold', fontsize=12)
    ax.set_xlabel('Correlation Coefficient')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('figures/04_correlation.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: figures/04_correlation.png")
    plt.close()

def plot_demand_by_season(df):
    """Plot demand patterns by season"""
    print("\n🌡️  Creating seasonal demand plot...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    season_names = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'}
    colors = {'Winter': '#3b82f6', 'Spring': '#10b981', 'Summer': '#ef4444', 'Autumn': '#f59e0b'}
    
    for season in [1, 2, 3, 4]:
        season_data = df[df['Season'] == season]['Electric_demand']
        ax.hist(season_data, bins=30, alpha=0.6, label=season_names[season], color=colors[season_names[season]])
    
    ax.set_title('Electricity Demand Distribution by Season', fontweight='bold', fontsize=12)
    ax.set_xlabel('Demand (MW)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('figures/05_seasonal_demand.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: figures/05_seasonal_demand.png")
    plt.close()

def summary_statistics(df):
    """Display summary statistics"""
    print("\n📋 Summary Statistics:")
    print("-" * 60)
    print(df[['Temperature', 'Humidity', 'Wind_speed', 'PV_production', 'Wind_production', 'Electric_demand']].describe().round(2))

if __name__ == "__main__":
    create_figures_dir()
    df = load_data()
    
    print("\n🔍 Running analysis...")
    plot_time_series(df)
    plot_seasonal_decomposition(df)
    plot_acf_pacf(df)
    adf_test(df)
    correlation_analysis(df)
    plot_demand_by_season(df)
    summary_statistics(df)
    
    print("\n" + "=" * 80)
    print("✅ EDA COMPLETE!")
    print("=" * 80)
    print("\n📁 Check 'figures/' folder for visualizations")
    print("\n📌 Next Step: Build SARIMA model")
    print("   Run: python sarima_model.py")