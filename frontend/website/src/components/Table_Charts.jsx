import React, { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend } from "recharts";

export default function Table_Charts() {
  const [weatherData, setWeatherData] = useState([]);
  const [forecastData, setForecastData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [city, setCity] = useState(null);
  const [inputCity, setInputCity] = useState("");
  const [activeTab, setActiveTab] = useState("forecast");
  const [unit, setUnit] = useState("C"); // "C" for Celsius, "F" for Fahrenheit

  // Convert Celsius to Fahrenheit
  const toF = (celsius) => Math.round((celsius * 9/5 + 32) * 10) / 10;
  
  // Display temperature based on selected unit
  const displayTemp = (celsius) => {
    if (unit === "F") {
      return `${toF(celsius)}°F`;
    }
    return `${celsius}°C`;
  };
  
  // Display just the number (for charts/tables)
  const displayTempValue = (celsius) => {
    return unit === "F" ? toF(celsius) : celsius;
  };

  const BACKEND_URL = "http://localhost:8000";

  const fetchWeather = async (cityName) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/weather?city=${cityName}`);
      if (!res.ok) throw new Error("Failed to fetch weather data");
      const data = await res.json();

      if (data.error) {
        throw new Error(data.error);
      }

      setWeatherData([
        {
          datetime: new Date().toLocaleString(),
          temp: data.temperature,
          feels_like: data.feels_like,
          humidity: data.humidity,
          weather: data.weather_desc,
          wind: data.wind_speed,
        },
      ]);
      setCity(cityName);
      generateMockForecast(data.temperature);
      setLoading(false);
    } catch (err) {
      console.error(err);
      generateDemoData(cityName);
      setLoading(false);
    }
  };

  const generateDemoData = (cityName) => {
    const baseTemp = 15 + Math.random() * 15;
    setWeatherData([
      {
        datetime: new Date().toLocaleString(),
        temp: Math.round(baseTemp * 10) / 10,
        feels_like: Math.round((baseTemp - 2) * 10) / 10,
        humidity: Math.round(40 + Math.random() * 40),
        weather: ["Sunny", "Partly Cloudy", "Cloudy", "Clear"][Math.floor(Math.random() * 4)],
        wind: Math.round((2 + Math.random() * 8) * 10) / 10,
      },
    ]);
    setCity(cityName);
    generateMockForecast(baseTemp);
  };

  const generateMockForecast = (baseTemp) => {
    const forecast = [];
    const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    const today = new Date();
    
    for (let i = 0; i < 7; i++) {
      const date = new Date(today);
      date.setDate(date.getDate() + i);
      const variation = (Math.random() - 0.5) * 8;
      const temp = Math.round((baseTemp + variation) * 10) / 10;
      
      forecast.push({
        day: days[date.getDay()],
        date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        high: Math.round((temp + 3 + Math.random() * 2) * 10) / 10,
        low: Math.round((temp - 3 - Math.random() * 2) * 10) / 10,
        temp: temp,
        humidity: Math.round(40 + Math.random() * 40),
      });
    }
    setForecastData(forecast);
  };

  const handleSearch = () => {
    if (inputCity.trim()) {
      fetchWeather(inputCity);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  const quickCities = ["Tokyo", "New York", "London", "Los Angeles", "Sydney", "Paris"];

  // Landing state - no city selected
  if (!city) {
    return (
      <div style={{ padding: "2rem", color: "#fff", display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center" }}>
        <div style={{ textAlign: "center", maxWidth: "500px" }}>
          <p style={{ marginBottom: "2rem", fontSize: "1rem", color: "#94a3b8" }}>
            Search for any city to get current conditions and forecasts
          </p>
          
          <div style={{ display: "flex", gap: "10px", marginBottom: "1.5rem" }}>
            <input
              type="text"
              value={inputCity}
              onChange={(e) => setInputCity(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Enter city name..."
              style={{
                padding: "12px 16px",
                borderRadius: "8px",
                border: "1px solid #333",
                backgroundColor: "#1e1e1e",
                color: "#fff",
                flex: 1,
                fontSize: "1rem",
                outline: "none",
              }}
            />
            <button
              onClick={handleSearch}
              style={{
                padding: "12px 24px",
                background: "linear-gradient(135deg, #f59e0b, #d97706)",
                color: "#fff",
                border: "none",
                borderRadius: "8px",
                cursor: "pointer",
                fontWeight: "600",
                fontSize: "1rem",
              }}
            >
              Search
            </button>
          </div>

          <p style={{ fontSize: "0.85rem", color: "#666", marginBottom: "0.75rem" }}>Quick select:</p>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", justifyContent: "center" }}>
            {quickCities.map((c) => (
              <button
                key={c}
                onClick={() => fetchWeather(c)}
                style={{
                  padding: "0.5rem 1rem",
                  backgroundColor: "#1e1e1e",
                  color: "#ccc",
                  border: "1px solid #333",
                  borderRadius: "6px",
                  cursor: "pointer",
                  fontSize: "0.85rem",
                  transition: "all 0.2s",
                }}
              >
                {c}
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div style={{ padding: "2rem", textAlign: "center", color: "#ccc", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <div>
          <div style={{ fontSize: "2rem", marginBottom: "1rem" }}>🌤️</div>
          <p>Loading weather data...</p>
        </div>
      </div>
    );
  }

  return (
    <div style={{ padding: "1rem 0", color: "#fff" }}>
      <div style={{ maxWidth: "1200px", margin: "0 auto" }}>
        
        {/* Search Bar */}
        <div style={{ display: "flex", gap: "10px", justifyContent: "center", marginBottom: "1rem" }}>
          <input
            type="text"
            value={inputCity}
            onChange={(e) => setInputCity(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Search another city..."
            style={{
              padding: "10px 14px",
              borderRadius: "6px",
              border: "1px solid #333",
              backgroundColor: "#1e1e1e",
              color: "#fff",
              width: "250px",
              fontSize: "0.95rem",
            }}
          />
          <button
            onClick={handleSearch}
            style={{
              padding: "10px 20px",
              background: "linear-gradient(135deg, #f59e0b, #d97706)",
              color: "#fff",
              border: "none",
              borderRadius: "6px",
              cursor: "pointer",
              fontWeight: "600",
            }}
          >
            Search
          </button>
        </div>

        {/* Quick Cities & Unit Toggle */}
        <div style={{ display: "flex", gap: "1.5rem", justifyContent: "center", alignItems: "center", flexWrap: "wrap", marginBottom: "1.5rem" }}>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", justifyContent: "center" }}>
            {quickCities.map((c) => (
              <button
                key={c}
                onClick={() => fetchWeather(c)}
                style={{
                  padding: "0.35rem 0.75rem",
                  backgroundColor: city === c ? "#f59e0b" : "#1e1e1e",
                  color: city === c ? "#000" : "#999",
                  border: "1px solid #333",
                  borderRadius: "4px",
                  cursor: "pointer",
                  fontSize: "0.8rem",
                  fontWeight: city === c ? "600" : "normal",
                }}
              >
                {c}
              </button>
            ))}
          </div>

          {/* Unit Toggle */}
          <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
            <span style={{ color: "#666", fontSize: "0.85rem" }}>Unit:</span>
            <button
              onClick={() => setUnit("C")}
              style={{
                padding: "0.35rem 0.75rem",
                backgroundColor: unit === "C" ? "#f59e0b" : "#1e1e1e",
                color: unit === "C" ? "#000" : "#999",
                border: "1px solid #333",
                borderRadius: "4px",
                cursor: "pointer",
                fontSize: "0.8rem",
                fontWeight: unit === "C" ? "600" : "normal",
              }}
            >
              °C
            </button>
            <button
              onClick={() => setUnit("F")}
              style={{
                padding: "0.35rem 0.75rem",
                backgroundColor: unit === "F" ? "#f59e0b" : "#1e1e1e",
                color: unit === "F" ? "#000" : "#999",
                border: "1px solid #333",
                borderRadius: "4px",
                cursor: "pointer",
                fontSize: "0.8rem",
                fontWeight: unit === "F" ? "600" : "normal",
              }}
            >
              °F
            </button>
          </div>
        </div>

        {error && (
          <p style={{ color: "#ef4444", textAlign: "center", marginBottom: "1rem", padding: "1rem", backgroundColor: "rgba(239, 68, 68, 0.1)", borderRadius: "8px" }}>
            ⚠️ {error}
          </p>
        )}

        {/* Current Weather Card */}
        {weatherData.length > 0 && (
          <div style={{ backgroundColor: "#1e1e1e", padding: "2rem", borderRadius: "12px", marginBottom: "2rem", border: "1px solid #333" }}>
            <h2 style={{ marginBottom: "1.5rem", textAlign: "center", fontSize: "1.5rem" }}>
              📍 {city}
            </h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "1.5rem", textAlign: "center" }}>
              <div style={{ backgroundColor: "#252525", padding: "1.25rem", borderRadius: "8px" }}>
                <p style={{ fontSize: "0.8rem", color: "#999", marginBottom: "0.5rem" }}>🌡️ Temperature</p>
                <p style={{ fontSize: "2rem", fontWeight: "bold", color: "#f59e0b" }}>
                  {displayTemp(weatherData[0].temp)}
                </p>
              </div>
              <div style={{ backgroundColor: "#252525", padding: "1.25rem", borderRadius: "8px" }}>
                <p style={{ fontSize: "0.8rem", color: "#999", marginBottom: "0.5rem" }}>🤔 Feels Like</p>
                <p style={{ fontSize: "2rem", fontWeight: "bold", color: "#eab308" }}>
                  {displayTemp(weatherData[0].feels_like)}
                </p>
              </div>
              <div style={{ backgroundColor: "#252525", padding: "1.25rem", borderRadius: "8px" }}>
                <p style={{ fontSize: "0.8rem", color: "#999", marginBottom: "0.5rem" }}>💧 Humidity</p>
                <p style={{ fontSize: "2rem", fontWeight: "bold", color: "#10b981" }}>
                  {weatherData[0].humidity}%
                </p>
              </div>
              <div style={{ backgroundColor: "#252525", padding: "1.25rem", borderRadius: "8px" }}>
                <p style={{ fontSize: "0.8rem", color: "#999", marginBottom: "0.5rem" }}>💨 Wind Speed</p>
                <p style={{ fontSize: "2rem", fontWeight: "bold", color: "#fbbf24" }}>
                  {weatherData[0].wind} <span style={{ fontSize: "1rem" }}>m/s</span>
                </p>
              </div>
            </div>
            <p style={{ marginTop: "1.5rem", textAlign: "center", fontSize: "1.1rem", color: "#ccc" }}>
              ☁️ {weatherData[0].weather}
            </p>
            <p style={{ textAlign: "center", fontSize: "0.8rem", color: "#666", marginTop: "0.5rem" }}>
              Last updated: {weatherData[0].datetime}
            </p>
          </div>
        )}

        {/* Tabs */}
        <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem", borderBottom: "1px solid #333", paddingBottom: "0.5rem" }}>
          {[
            { id: "forecast", label: "📅 7-Day Forecast" },
            { id: "chart", label: "📈 Temperature Chart" },
            { id: "table", label: "📋 Data Table" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: "0.75rem 1.25rem",
                borderRadius: "6px 6px 0 0",
                border: "none",
                backgroundColor: activeTab === tab.id ? "#1e1e1e" : "transparent",
                color: activeTab === tab.id ? "#fff" : "#666",
                cursor: "pointer",
                fontSize: "0.9rem",
                fontWeight: activeTab === tab.id ? "600" : "normal",
                borderBottom: activeTab === tab.id ? "2px solid #f59e0b" : "none",
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div style={{ backgroundColor: "#1e1e1e", padding: "1.5rem", borderRadius: "8px", border: "1px solid #333" }}>
          
          {/* 7-Day Forecast Tab */}
          {activeTab === "forecast" && forecastData.length > 0 && (
            <div>
              <h3 style={{ marginBottom: "1rem", fontSize: "1.1rem" }}>7-Day Forecast for {city}</h3>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: "1rem" }}>
                {forecastData.map((day, idx) => (
                  <div 
                    key={idx} 
                    style={{ 
                      backgroundColor: idx === 0 ? "rgba(245, 158, 11, 0.15)" : "#252525", 
                      padding: "1rem", 
                      borderRadius: "8px", 
                      textAlign: "center",
                      border: idx === 0 ? "1px solid #f59e0b" : "1px solid #333",
                    }}
                  >
                    <p style={{ fontWeight: "bold", color: idx === 0 ? "#f59e0b" : "#ccc", marginBottom: "0.25rem" }}>
                      {idx === 0 ? "Today" : day.day}
                    </p>
                    <p style={{ fontSize: "0.75rem", color: "#666", marginBottom: "0.75rem" }}>{day.date}</p>
                    <p style={{ fontSize: "1.5rem", fontWeight: "bold", color: "#fff" }}>{displayTempValue(day.temp)}°</p>
                    <div style={{ display: "flex", justifyContent: "center", gap: "0.5rem", marginTop: "0.5rem", fontSize: "0.8rem" }}>
                      <span style={{ color: "#ef4444" }}>↑{displayTempValue(day.high)}°</span>
                      <span style={{ color: "#10b981" }}>↓{displayTempValue(day.low)}°</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Chart Tab */}
          {activeTab === "chart" && forecastData.length > 0 && (
            <div>
              <h3 style={{ marginBottom: "1rem", fontSize: "1.1rem" }}>Temperature Trend ({unit === "F" ? "°F" : "°C"})</h3>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart 
                  data={forecastData.map(d => ({
                    ...d,
                    highDisplay: displayTempValue(d.high),
                    tempDisplay: displayTempValue(d.temp),
                    lowDisplay: displayTempValue(d.low),
                  }))} 
                  margin={{ top: 10, right: 30, left: 0, bottom: 10 }}
                >
                  <CartesianGrid stroke="#333" strokeDasharray="3 3" />
                  <XAxis dataKey="day" tick={{ fontSize: 12, fill: "#999" }} />
                  <YAxis tick={{ fontSize: 12, fill: "#999" }} domain={['dataMin - 5', 'dataMax + 5']} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: "#252525", border: "1px solid #444", borderRadius: "6px" }}
                    formatter={(value) => [`${value}°${unit}`, ""]}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="highDisplay" stroke="#ef4444" name="High" strokeWidth={2} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="tempDisplay" stroke="#f59e0b" name="Avg" strokeWidth={2} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="lowDisplay" stroke="#10b981" name="Low" strokeWidth={2} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Table Tab */}
          {activeTab === "table" && forecastData.length > 0 && (
            <div style={{ overflowX: "auto" }}>
              <h3 style={{ marginBottom: "1rem", fontSize: "1.1rem" }}>Detailed Forecast Data</h3>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
                <thead>
                  <tr style={{ backgroundColor: "#252525", borderBottom: "2px solid #444" }}>
                    <th style={{ padding: "12px", textAlign: "left" }}>Day</th>
                    <th style={{ padding: "12px", textAlign: "left" }}>Date</th>
                    <th style={{ padding: "12px", textAlign: "center" }}>High (°{unit})</th>
                    <th style={{ padding: "12px", textAlign: "center" }}>Avg (°{unit})</th>
                    <th style={{ padding: "12px", textAlign: "center" }}>Low (°{unit})</th>
                    <th style={{ padding: "12px", textAlign: "center" }}>Humidity (%)</th>
                  </tr>
                </thead>
                <tbody>
                  {forecastData.map((day, idx) => (
                    <tr key={idx} style={{ borderBottom: "1px solid #333", backgroundColor: idx % 2 === 0 ? "#1a1a1a" : "#1e1e1e" }}>
                      <td style={{ padding: "10px", fontWeight: idx === 0 ? "bold" : "normal", color: idx === 0 ? "#f59e0b" : "#ccc" }}>
                        {idx === 0 ? "Today" : day.day}
                      </td>
                      <td style={{ padding: "10px", color: "#999" }}>{day.date}</td>
                      <td style={{ padding: "10px", textAlign: "center", color: "#ef4444", fontWeight: "500" }}>{displayTempValue(day.high)}</td>
                      <td style={{ padding: "10px", textAlign: "center", color: "#f59e0b", fontWeight: "bold" }}>{displayTempValue(day.temp)}</td>
                      <td style={{ padding: "10px", textAlign: "center", color: "#10b981", fontWeight: "500" }}>{displayTempValue(day.low)}</td>
                      <td style={{ padding: "10px", textAlign: "center", color: "#fbbf24" }}>{day.humidity}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}