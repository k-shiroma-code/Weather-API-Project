import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 6)

def load_data():
    """Load and prepare data"""
    print("\n" + "=" * 80)
    print("🔌 CALIFORNIA GRID LOAD FORECASTING - SARIMA MODEL")
    print("=" * 80)
    
    df = pd.read_csv('data/processed/california_data_processed.csv')
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    
    # Use daily averages for SARIMA (reduces noise, more stable)
    df_daily = df.set_index('Time')['Electric_demand'].resample('D').mean()
    
    print(f"\n✅ Data loaded: {len(df):,} hourly observations")
    print(f"📊 Converted to daily: {len(df_daily)} observations")
    print(f"📅 Date range: {df_daily.index.min()} to {df_daily.index.max()}")
    
    return df_daily

def split_train_test(data, test_size=90):
    """Split data into train and test sets"""
    print(f"\n📊 Train/Test Split:")
    print(f"   Test size: {test_size} days")
    
    train = data[:-test_size]
    test = data[-test_size:]
    
    print(f"   Train: {len(train)} days ({train.index.min()} to {train.index.max()})")
    print(f"   Test:  {len(test)} days ({test.index.min()} to {test.index.max()})")
    
    return train, test

def find_optimal_parameters(train_data):
    """Use auto_arima to find optimal SARIMA parameters"""
    print(f"\n🔍 Finding optimal SARIMA parameters...")
    print(f"   This may take a few minutes...")
    
    # auto_arima with minimal search space for speed
    # m=7 for weekly seasonality (much faster than yearly)
    auto_model = auto_arima(
        train_data,
        seasonal=True,
        m=7,  # weekly seasonality instead of 365
        start_p=0,
        start_q=0,
        start_P=0,
        start_Q=0,
        max_p=1,
        max_q=1,
        max_P=0,
        max_Q=0,
        max_d=1,
        max_D=1,
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    
    print(f"\n✅ Optimal parameters found:")
    print(f"   SARIMA{auto_model.order}{auto_model.seasonal_order}")
    print(f"   AIC: {auto_model.aic():.2f}")
    
    return auto_model.order, auto_model.seasonal_order

def build_sarima_model(train_data, order, seasonal_order):
    """Build and fit SARIMA model"""
    print(f"\n📈 Building SARIMA{order}{seasonal_order} model...")
    
    model = SARIMAX(
        train_data,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    results = model.fit(disp=False)
    
    print(f"✅ Model fitted successfully")
    print(f"\n{results.summary()}")
    
    return results

def forecast_and_evaluate(results, train_data, test_data):
    """Forecast on test set and evaluate"""
    print(f"\n🔮 Forecasting on test set ({len(test_data)} days)...")
    
    # Get forecast
    forecast = results.get_forecast(steps=len(test_data))
    forecast_df = forecast.conf_int()
    forecast_df['forecast'] = forecast.predicted_mean
    
    predictions = forecast.predicted_mean.values
    
    # Calculate metrics
    mae = mean_absolute_error(test_data, predictions)
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    mape = mean_absolute_percentage_error(test_data, predictions)
    
    print(f"\n📊 Forecast Metrics:")
    print(f"   MAE (Mean Absolute Error):           {mae:,.2f} MW")
    print(f"   RMSE (Root Mean Squared Error):      {rmse:,.2f} MW")
    print(f"   MAPE (Mean Absolute % Error):        {mape:.2f}%")
    
    # Baseline comparison (naive forecast = last value repeated)
    baseline_pred = np.full(len(test_data), test_data.iloc[0])
    baseline_mae = mean_absolute_error(test_data, baseline_pred)
    print(f"\n📌 Baseline (Naive Forecast) MAE:     {baseline_mae:,.2f} MW")
    improvement = ((baseline_mae - mae) / baseline_mae) * 100
    print(f"   Improvement over baseline:           {improvement:.2f}%")
    
    return predictions, forecast_df, mae, rmse, mape

def plot_forecast(train_data, test_data, predictions, forecast_df):
    """Plot actual vs forecasted demand"""
    print(f"\n📉 Creating forecast plot...")
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Train data
    ax.plot(train_data.index, train_data.values, label='Training Data', color='#1f77b4', linewidth=2)
    
    # Test data
    ax.plot(test_data.index, test_data.values, label='Actual Test Data', color='#2ca02c', linewidth=2)
    
    # Forecast
    ax.plot(test_data.index, predictions, label='SARIMA Forecast', color='#ff7f0e', linewidth=2, linestyle='--')
    
    # Confidence interval
    ax.fill_between(
        test_data.index,
        forecast_df.iloc[:, 0].values,
        forecast_df.iloc[:, 1].values,
        alpha=0.2,
        color='#ff7f0e',
        label='95% Confidence Interval'
    )
    
    ax.set_title('SARIMA Electricity Demand Forecast - California Grid', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Electricity Demand (MW)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/05_sarima_forecast.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: figures/05_sarima_forecast.png")
    plt.close()

def plot_residuals(results):
    """Plot residual diagnostics"""
    print(f"\n📊 Creating residual diagnostics plot...")
    
    fig = results.plot_diagnostics(figsize=(14, 8))
    plt.tight_layout()
    plt.savefig('figures/06_residual_diagnostics.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: figures/06_residual_diagnostics.png")
    plt.close()

def plot_error_analysis(test_data, predictions):
    """Analyze forecast errors"""
    print(f"\n📈 Creating error analysis plot...")
    
    errors = test_data.values - predictions
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Error over time
    axes[0, 0].plot(test_data.index, errors, color='#d62728', linewidth=1)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[0, 0].set_title('Forecast Error Over Time', fontweight='bold')
    axes[0, 0].set_ylabel('Error (MW)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Error distribution
    axes[0, 1].hist(errors, bins=30, color='#1f77b4', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Distribution of Forecast Errors', fontweight='bold')
    axes[0, 1].set_xlabel('Error (MW)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Actual vs Predicted
    axes[1, 0].scatter(test_data.values, predictions, alpha=0.5, color='#2ca02c')
    axes[1, 0].plot([test_data.min(), test_data.max()], [test_data.min(), test_data.max()], 'r--', linewidth=2)
    axes[1, 0].set_title('Actual vs Predicted', fontweight='bold')
    axes[1, 0].set_xlabel('Actual Demand (MW)')
    axes[1, 0].set_ylabel('Predicted Demand (MW)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Absolute percentage error
    ape = np.abs((test_data.values - predictions) / test_data.values) * 100
    axes[1, 1].plot(test_data.index, ape, color='#ff7f0e', linewidth=1)
    axes[1, 1].set_title('Absolute Percentage Error', fontweight='bold')
    axes[1, 1].set_ylabel('APE (%)')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/07_error_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: figures/07_error_analysis.png")
    plt.close()

if __name__ == "__main__":
    # Load data
    data = load_data()
    
    # Split data
    train, test = split_train_test(data, test_size=90)
    
    # Find optimal parameters
    order, seasonal_order = find_optimal_parameters(train)
    
    # Build model
    results = build_sarima_model(train, order, seasonal_order)
    
    # Forecast and evaluate
    predictions, forecast_df, mae, rmse, mape = forecast_and_evaluate(results, train, test)
    
    # Create visualizations
    plot_forecast(train, test, predictions, forecast_df)
    plot_residuals(results)
    plot_error_analysis(test, predictions)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ SARIMA MODELING COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Final Results:")
    print(f"   Model: SARIMA{order}{seasonal_order}")
    print(f"   MAE: {mae:,.2f} MW")
    print(f"   RMSE: {rmse:,.2f} MW")
    print(f"   MAPE: {mape:.2f}%")
    print(f"\n📁 Check 'figures/' for visualizations")
    print(f"\n💡 Next steps:")
    print(f"   - Tune hyperparameters for better accuracy")
    print(f"   - Try exogenous variables (temperature, solar production)")
    print(f"   - Build ensemble models")