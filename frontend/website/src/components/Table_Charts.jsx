import React, { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from "recharts";

export default function Table_Charts() {
  const [weatherData, setWeatherData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [city, setCity] = useState(null);
  const [inputCity, setInputCity] = useState("");

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

      // Format data for charts/tables
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
      setLoading(false);
    } catch (err) {
      console.error(err);
      setError(err.message);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchWeather(city);
  }, []);

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

  if (!city) {
    return (
      <div
        style={{
          padding: "2rem",
          color: "#fff",
          backgroundColor: "#121212",
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <h1 style={{ marginBottom: "2rem", fontSize: "2rem" }}>
          Weather Dashboard
        </h1>
        <p style={{ marginBottom: "1.5rem", fontSize: "1.1rem", color: "#ccc" }}>
          Enter a city name to get started
        </p>
        <div style={{ display: "flex", gap: "10px" }}>
          <input
            type="text"
            value={inputCity}
            onChange={(e) => setInputCity(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Enter city name..."
            style={{
              padding: "10px",
              borderRadius: "4px",
              border: "1px solid #555",
              backgroundColor: "#222",
              color: "#fff",
              width: "300px",
            }}
          />
          <button
            onClick={handleSearch}
            style={{
              padding: "10px 20px",
              backgroundColor: "#3b82f6",
              color: "#fff",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer",
              fontWeight: "bold",
            }}
          >
            Search
          </button>
        </div>
      </div>
    );
  }

  if (loading)
    return (
      <p style={{ color: "#ccc", textAlign: "center", paddingTop: "2rem" }}>
        Loading...
      </p>
    );

  return (
    <div
      style={{
        padding: "2rem",
        color: "#fff",
        backgroundColor: "#121212",
        minHeight: "100vh",
      }}
    >
      <h1 style={{ textAlign: "center", marginBottom: "2rem", fontSize: "2rem" }}>
        Weather Dashboard
      </h1>

      {/* Search Bar */}
      <div style={{ textAlign: "center", marginBottom: "2rem" }}>
        <div style={{ display: "flex", gap: "10px", justifyContent: "center" }}>
          <input
            type="text"
            value={inputCity}
            onChange={(e) => setInputCity(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Enter city name..."
            style={{
              padding: "10px",
              borderRadius: "4px",
              border: "1px solid #555",
              backgroundColor: "#222",
              color: "#fff",
              width: "250px",
            }}
          />
          <button
            onClick={handleSearch}
            style={{
              padding: "10px 20px",
              backgroundColor: "#3b82f6",
              color: "#fff",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer",
              fontWeight: "bold",
            }}
          >
            Search
          </button>
        </div>
      </div>

      {error && (
        <p style={{ color: "#ef4444", textAlign: "center", marginBottom: "1rem" }}>
          Error: {error}
        </p>
      )}

      {/* Current Weather Info */}
      {weatherData.length > 0 && (
        <div
          style={{
            backgroundColor: "#222",
            padding: "1.5rem",
            borderRadius: "8px",
            marginBottom: "2rem",
            textAlign: "center",
          }}
        >
          <h2 style={{ marginBottom: "1rem" }}>{city}</h2>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
              gap: "1rem",
            }}
          >
            <div>
              <p style={{ fontSize: "0.9rem", color: "#999" }}>Temperature</p>
              <p style={{ fontSize: "1.5rem", fontWeight: "bold" }}>
                {weatherData[0].temp}°C
              </p>
            </div>
            <div>
              <p style={{ fontSize: "0.9rem", color: "#999" }}>Feels Like</p>
              <p style={{ fontSize: "1.5rem", fontWeight: "bold" }}>
                {weatherData[0].feels_like}°C
              </p>
            </div>
            <div>
              <p style={{ fontSize: "0.9rem", color: "#999" }}>Humidity</p>
              <p style={{ fontSize: "1.5rem", fontWeight: "bold" }}>
                {weatherData[0].humidity}%
              </p>
            </div>
            <div>
              <p style={{ fontSize: "0.9rem", color: "#999" }}>Wind Speed</p>
              <p style={{ fontSize: "1.5rem", fontWeight: "bold" }}>
                {weatherData[0].wind} m/s
              </p>
            </div>
          </div>
          <p style={{ marginTop: "1rem", fontSize: "1.1rem", textTransform: "capitalize" }}>
            {weatherData[0].weather}
          </p>
        </div>
      )}

      {/* Table */}
      <div style={{ overflowX: "auto", marginBottom: "2rem" }}>
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            backgroundColor: "#222",
          }}
        >
          <thead>
            <tr style={{ backgroundColor: "#333" }}>
              <th style={{ padding: "10px", textAlign: "left" }}>Datetime</th>
              <th style={{ padding: "10px", textAlign: "left" }}>Temp (°C)</th>
              <th style={{ padding: "10px", textAlign: "left" }}>Feels Like</th>
              <th style={{ padding: "10px", textAlign: "left" }}>Humidity (%)</th>
              <th style={{ padding: "10px", textAlign: "left" }}>Weather</th>
              <th style={{ padding: "10px", textAlign: "left" }}>Wind (m/s)</th>
            </tr>
          </thead>
          <tbody>
            {weatherData.map((w, idx) => (
              <tr key={idx} style={{ borderBottom: "1px solid #333" }}>
                <td style={{ padding: "8px" }}>{w.datetime}</td>
                <td style={{ padding: "8px" }}>{w.temp}</td>
                <td style={{ padding: "8px" }}>{w.feels_like}</td>
                <td style={{ padding: "8px" }}>{w.humidity}</td>
                <td style={{ padding: "8px", textTransform: "capitalize" }}>
                  {w.weather}
                </td>
                <td style={{ padding: "8px" }}>{w.wind}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Temperature Chart */}
      {weatherData.length > 0 && (
        <div style={{ background: "#222", padding: "1rem", borderRadius: "8px" }}>
          <h3 style={{ textAlign: "center", marginBottom: "1rem" }}>
            Weather Summary
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={weatherData}>
              <CartesianGrid stroke="#444" />
              <XAxis dataKey="datetime" tick={{ fontSize: 12 }} />
              <YAxis />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="temp"
                stroke="#3b82f6"
                name="Temperature"
              />
              <Line
                type="monotone"
                dataKey="feels_like"
                stroke="#ef4444"
                name="Feels Like"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}