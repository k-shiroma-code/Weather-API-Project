import requests
import pandas as pd
import os
from datetime import datetime, timedelta

def fetch_tokyo_weather():
    """
    Fetch historical weather data from Open-Meteo Historical Forecast API for Tokyo
    Saves raw data to CSV file
    """
    print("🌍 Fetching historical weather data for Tokyo...")
    
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    
    # Fetch data in smaller chunks to avoid timeout
    all_data = []
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 23)
    current_date = start_date
    
    # Fetch in 3-month chunks
    chunk_size = 90
    
    try:
        while current_date < end_date:
            chunk_end = min(current_date + timedelta(days=chunk_size), end_date)
            
            print(f"Fetching {current_date.date()} to {chunk_end.date()}...")
            
            params = {
                "latitude": 35.6895,
                "longitude": 139.6917,
                "start_date": current_date.strftime("%Y-%m-%d"),
                "end_date": chunk_end.strftime("%Y-%m-%d"),
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
                "timezone": "Asia/Tokyo"
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            df_chunk = pd.DataFrame({
                "date": pd.to_datetime(data["daily"]["time"]),
                "temp_max": data["daily"]["temperature_2m_max"],
                "temp_min": data["daily"]["temperature_2m_min"],
                "precipitation": data["daily"]["precipitation_sum"],
                "wind_speed_max": data["daily"]["wind_speed_10m_max"]
            })
            
            all_data.append(df_chunk)
            current_date = chunk_end + timedelta(days=1)
        
        # Combine all chunks
        df = pd.concat(all_data, ignore_index=True)
        
        # Calculate mean temperature
        df["temp_mean"] = (df["temp_max"] + df["temp_min"]) / 2
        
        # Set date as index
        df.set_index("date", inplace=True)
        
        # Create directory if it doesn't exist
        os.makedirs("data/raw", exist_ok=True)
        
        # Save raw data
        csv_path = "data/raw/tokyo_weather_raw.csv"
        df.to_csv(csv_path)
        
        print(f"\n✅ Data fetched successfully!")
        print(f"📊 Shape: {df.shape}")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        print(f"💾 Saved to: {csv_path}")
        print(f"\nFirst 5 rows:\n{df.head()}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

if __name__ == "__main__":
    fetch_tokyo_weather()