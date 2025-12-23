# 🌤️ Weather API Project

A full-stack weather forecasting application with real-time data and AI-powered SARIMA predictions for Tokyo.

## 📊 Features

- **Real-time Weather Data** - Current conditions for any city via OpenWeather API
- **SARIMA ML Forecasting** - 14-day temperature predictions using 5 years of historical data
- **Interactive Dashboard** - Beautiful React/Astro frontend with charts and tables
- **REST API** - FastAPI backend with multiple endpoints
- **Historical Data Pipeline** - ETL pipeline fetching from Open-Meteo API

## 🏗️ Architecture

```
weather-api-project/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx
│   │   │   ├── Table_Charts.jsx
│   │   │   ├── Forecast.jsx
│   │   │   └── SarimaForecast.jsx
│   │   └── pages/
│   │       ├── index.astro
│   │       ├── table_charts.astro
│   │       └── forecast.astro
│   └── package.json
│
├── backend/
│   ├── api/
│   │   └── table_charts_main.py
│   ├── ml/
│   │   ├── data/
│   │   │   ├── raw/
│   │   │   └── processed/
│   │   ├── models/
│   │   ├── figures/
│   │   ├── fetch_data.py
│   │   ├── eda.py
│   │   ├── train_model.py
│   │   └── predict.py
│   ├── app.py
│   └── requirements.txt
│
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- Git

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend/website
npm install
npm run dev
```

## 📈 ML Pipeline

### 1. Fetch Data
Fetches 5 years of historical weather data for Tokyo (2020-2024) from Open-Meteo API:

```bash
cd backend/ml
python fetch_data.py
```

**Output:** `data/raw/tokyo_weather_raw.csv` (1819 records)

### 2. Exploratory Data Analysis
Analyzes data, creates visualizations, tests stationarity:

```bash
python eda.py
```

**Output:** 
- Cleaned data: `data/processed/tokyo_weather_processed.csv`
- 4 visualization PNG files in `figures/`

### 3. Train SARIMA Model
Finds optimal SARIMA parameters and trains the model (20-40 minutes):

```bash
python train_model.py
```

**Output:**
- Trained model: `models/sarima_model.pkl`
- Parameters: `models/sarima_params.pkl`
- Results plot: `figures/04_sarima_results.png`

### 4. Make Predictions
Generate 14-day temperature forecasts:

```bash
python predict.py
```

## 🔌 API Endpoints

### Weather Endpoints

**Get current weather for a city:**
```bash
GET /weather?city=Tokyo
```

Response:
```json
{
  "city": "Tokyo",
  "temperature": 15.2,
  "feels_like": 14.1,
  "humidity": 65,
  "wind_speed": 7.5,
  "weather_desc": "partly cloudy"
}
```

### SARIMA Forecast Endpoints

**Get 14-day forecast:**
```bash
GET /sarima/forecast?days=14
```

Response:
```json
{
  "location": "Tokyo",
  "forecast_days": 14,
  "predictions": [
    {
      "day": 1,
      "date": "2025-01-01",
      "forecast": 12.5,
      "lower_ci": 10.2,
      "upper_ci": 14.8
    }
  ]
}
```

**Get model info:**
```bash
GET /sarima/info
```

## 🎨 Frontend Pages

- **Home** (`/`) - Landing page with project overview
- **Weather Dashboard** (`/table_charts`) - Real-time weather search and charts
- **ML Forecast** (`/forecast`) - SARIMA predictions with interactive charts

## 📊 Data Sources

- **Open-Meteo Historical Forecast API** - Historical weather data (2020-2024)
- **OpenWeather API** - Current weather conditions (any city)

## 🧠 Model Details

**SARIMA Parameters:**
- Automatically determined by `auto_arima`
- Typical: SARIMA(1,1,1)x(1,1,1,365)
- Training data: 1789 days (2020-11-23)
- Test data: 30 days (2024-11-24 to 2024-12-23)
- Evaluation metrics: RMSE, MAE, MAPE

**Features Used:**
- `temperature_2m_max` - Daily maximum temperature
- `temperature_2m_min` - Daily minimum temperature
- `precipitation_sum` - Daily precipitation
- `wind_speed_10m_max` - Maximum wind speed

## 📝 Environment Variables

Create a `.env` file in the backend directory:

```
OPENWEATHER_API_KEY=your_api_key_here
```

## 🛠️ Tech Stack

**Frontend:**
- Astro
- React
- Recharts
- Tailwind CSS

**Backend:**
- FastAPI
- Python 3.9+
- Pandas, NumPy
- Statsmodels (SARIMA)
- Scikit-learn

**Data Pipeline:**
- Open-Meteo API
- Pandas, NumPy
- Matplotlib, Seaborn
- PMDarima (auto_arima)

## 📦 Dependencies

See `backend/requirements.txt` for full list. Key packages:
- fastapi==0.127.0
- uvicorn==0.40.0
- pandas==2.3.3
- statsmodels==0.14.0
- pmdarima==2.0.4
- scikit-learn==1.7.2
- requests==2.32.5

## 🚀 Running the Project

**Terminal 1 - Backend API:**
```bash
cd backend
uvicorn app:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend/website
npm run dev
```

**Terminal 3 - ML Training (optional):**
```bash
cd backend/ml
python train_model.py
```

Visit http://localhost:4321 for the frontend and http://localhost:8000/docs for API docs.

## 📚 Learning Resources

- [SARIMA Overview](https://otexts.com/fpp2/arima.html)
- [Open-Meteo API Docs](https://open-meteo.com/en/docs)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/)
- [Astro Documentation](https://docs.astro.build/)

## 🤝 Contributing

Feel free to fork and submit PRs for improvements!

## 📄 License

MIT License - See LICENSE file for details

## 👤 Author

Kyle Shiroma - [GitHub](https://github.com/k-shiroma-code)

---

**Last Updated:** December 2024
