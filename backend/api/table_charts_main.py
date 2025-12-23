from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env file

app = FastAPI(title="Weather API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321"],  # frontend port
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Validate API key on startup
if not OPENWEATHER_API_KEY:
    raise ValueError("OPENWEATHER_API_KEY not found in environment variables")

@app.get("/")
def read_root():
    return {"status": "Weather API is running"}

@app.get("/weather")
def get_weather(city: str):
    """Fetch current weather for a given city"""
    
    # Validate city parameter
    if not city or not city.strip():
        raise HTTPException(status_code=400, detail="City parameter is required")
    
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city.strip()}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raise exception for bad status codes
        
        data = response.json()
        return {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "weather_desc": data["weather"][0]["description"],
        }
    
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Request to weather service timed out")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Unable to connect to weather service")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"City '{city}' not found")
        else:
            raise HTTPException(status_code=502, detail="Weather service error")
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred")