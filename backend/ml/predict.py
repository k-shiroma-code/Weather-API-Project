import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def load_model():
    """Load trained SARIMA model"""
    model_path = "models/sarima_model.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first!")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    return model

def load_params():
    """Load model parameters"""
    params_path = "models/sarima_params.pkl"
    
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Parameters not found at {params_path}")
    
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    
    return params

def predict_temperature(days=14):
    """
    Generate temperature predictions for next N days
    Returns DataFrame with predictions and confidence intervals
    """
    print(f"🔮 Generating {days}-day forecast...")
    
    try:
        model = load_model()
        params = load_params()
        
        # Get forecast
        forecast_result = model.get_forecast(steps=days)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.05)  # 95% confidence interval
        
        # Create result dataframe
        last_date = pd.to_datetime(model.fittedvalues.index[-1])
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')
        
        result_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecast_mean.values,
            'lower_ci': conf_int.iloc[:, 0].values,
            'upper_ci': conf_int.iloc[:, 1].values
        })
        
        result_df.set_index('date', inplace=True)
        
        print(f"✅ Forecast generated for {days} days")
        print(f"\nForecast Summary:")
        print(f"Location: {params['location']}")
        print(f"Date Range: {result_df.index[0].date()} to {result_df.index[-1].date()}")
        print(f"\nFirst 5 predictions:\n{result_df.head()}")
        
        return result_df
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None

def get_forecast_dict(days=14):
    """
    Get forecast as dictionary (for API responses)
    """
    forecast_df = predict_temperature(days)
    
    if forecast_df is None:
        return None
    
    params = load_params()
    
    forecast_list = []
    for idx, (date, row) in enumerate(forecast_df.iterrows()):
        forecast_list.append({
            "day": idx + 1,
            "date": date.strftime("%Y-%m-%d"),
            "forecast": float(row['forecast']),
            "lower_ci": float(row['lower_ci']),
            "upper_ci": float(row['upper_ci'])
        })
    
    return {
        "location": params['location'],
        "forecast_days": days,
        "data_range": params['data_range'],
        "order": str(params['order']),
        "seasonal_order": str(params['seasonal_order']),
        "predictions": forecast_list
    }

if __name__ == "__main__":
    # Test predictions
    forecast = predict_temperature(14)
    
    if forecast is not None:
        print("\n" + "=" * 60)
        print("Full Forecast DataFrame:")
        print(forecast)
        
        # Also show as dictionary format (for API)
        print("\n" + "=" * 60)
        forecast_dict = get_forecast_dict(14)
        print("Dictionary Format (for API):")
        import json
        print(json.dumps(forecast_dict, indent=2))