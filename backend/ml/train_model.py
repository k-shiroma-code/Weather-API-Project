import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

def load_processed_data():
    """Load the processed weather data"""
    print("📂 Loading processed data...")
    df = pd.read_csv("data/processed/tokyo_weather_processed.csv", 
                     index_col="date", parse_dates=True)
    print(f"✅ Loaded {len(df)} records")
    return df

def split_train_test(df, test_size=30):
    """Split data into train and test sets"""
    print(f"\n✂️  Splitting data (test size: {test_size} days)...")
    
    train = df[:-test_size]
    test = df[-test_size:]
    
    print(f"📊 Train set: {len(train)} records ({train.index.min()} to {train.index.max()})")
    print(f"📊 Test set: {len(test)} records ({test.index.min()} to {test.index.max()})")
    
    return train, test

def find_best_parameters(train_data):
    """Use auto_arima to find optimal SARIMA parameters"""
    print("\n🔍 Finding optimal SARIMA parameters (this may take a few minutes)...")
    print("⏳ Running auto_arima...")
    
    try:
        auto_model = auto_arima(
            train_data['temp_mean'],
            seasonal=True,
            m=365,  # Yearly seasonality
            max_p=3,
            max_d=2,
            max_q=3,
            max_P=2,
            max_D=1,
            max_Q=2,
            stepwise=True,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            information_criterion='aic'
        )
        
        print(f"\n✅ Best parameters found!")
        print(f"📈 Order (p,d,q): {auto_model.order}")
        print(f"📈 Seasonal Order (P,D,Q,m): {auto_model.seasonal_order}")
        print(f"📊 AIC: {auto_model.aic:.2f}")
        
        return auto_model.order, auto_model.seasonal_order
        
    except Exception as e:
        print(f"⚠️  Auto ARIMA failed: {e}")
        print("Using default parameters...")
        return (1, 1, 1), (1, 1, 1, 365)

def train_sarima(train_data, order, seasonal_order):
    """Train SARIMA model"""
    print(f"\n🤖 Training SARIMA model...")
    print(f"   Order: {order}")
    print(f"   Seasonal Order: {seasonal_order}")
    
    try:
        model = SARIMAX(
            train_data['temp_mean'],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted_model = model.fit(disp=False, maxiter=500)
        
        print(f"\n✅ Model trained successfully!")
        print(f"\n{fitted_model.summary()}")
        
        return fitted_model
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def evaluate_model(fitted_model, test_data):
    """Evaluate model on test set"""
    print(f"\n📊 Evaluating model on test set...")
    
    # Make predictions
    forecast = fitted_model.get_forecast(steps=len(test_data))
    predictions = forecast.predicted_mean
    
    # Calculate metrics
    actual = test_data['temp_mean'].values
    pred = predictions.values
    
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, pred)
    mape = mean_absolute_percentage_error(actual, pred)
    
    print(f"📈 Mean Squared Error (MSE): {mse:.4f}")
    print(f"📈 Root Mean Squared Error (RMSE): {rmse:.4f}°C")
    print(f"📈 Mean Absolute Error (MAE): {mae:.4f}°C")
    print(f"📈 Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
    
    return predictions, rmse, mae, mape

def plot_results(train_data, test_data, predictions, fitted_model):
    """Plot training results"""
    print(f"\n📊 Creating result plots...")
    
    os.makedirs("figures", exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Actual vs Predicted
    axes[0].plot(train_data.index, train_data['temp_mean'], label='Training Data', color='#3b82f6')
    axes[0].plot(test_data.index, test_data['temp_mean'], label='Actual Test Data', color='#10b981', linewidth=2)
    axes[0].plot(test_data.index, predictions, label='SARIMA Predictions', color='#ef4444', linestyle='--', linewidth=2)
    axes[0].set_title('SARIMA Model: Training & Test Data Predictions', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    residuals = test_data['temp_mean'].values - predictions.values
    axes[1].plot(test_data.index, residuals, marker='o', color='#f59e0b', linewidth=1, markersize=4)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_title('Prediction Residuals', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Residual (°C)')
    axes[1].set_xlabel('Date')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/04_sarima_results.png', dpi=100, bbox_inches='tight')
    print("✅ Saved: figures/04_sarima_results.png")
    plt.close()

def save_model(fitted_model, order, seasonal_order):
    """Save trained model"""
    print(f"\n💾 Saving model...")
    
    os.makedirs("models", exist_ok=True)
    
    model_path = "models/sarima_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(fitted_model, f)
    
    # Also save parameters
    params = {
        "order": order,
        "seasonal_order": seasonal_order,
        "location": "Tokyo",
        "data_range": "2022-01-01 to 2024-12-23"
    }
    
    params_path = "models/sarima_params.pkl"
    with open(params_path, "wb") as f:
        pickle.dump(params, f)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Parameters saved to: {params_path}")

def main():
    """Run complete training pipeline"""
    print("=" * 70)
    print("TRAINING SARIMA MODEL - TOKYO WEATHER FORECASTING")
    print("=" * 70)
    
    # Load data
    df = load_processed_data()
    
    # Split data
    train, test = split_train_test(df, test_size=30)
    
    # Find best parameters
    order, seasonal_order = find_best_parameters(train)
    
    # Train model
    fitted_model = train_sarima(train, order, seasonal_order)
    
    if fitted_model:
        # Evaluate model
        predictions, rmse, mae, mape = evaluate_model(fitted_model, test)
        
        # Plot results
        plot_results(train, test, predictions, fitted_model)
        
        # Save model
        save_model(fitted_model, order, seasonal_order)
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE!")
        print("=" * 70)
    else:
        print("❌ Training failed!")

if __name__ == "__main__":
    main()