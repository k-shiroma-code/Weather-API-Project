import React, { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, ComposedChart, Bar } from "recharts";

export default function Forecast() {
  const [forecast, setForecast] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [days, setDays] = useState(14);
  const [activeTab, setActiveTab] = useState("chart");

  const BACKEND_URL = "http://localhost:8000";

  useEffect(() => {
    fetchModelInfo();
    fetchForecast(14);
  }, []);

  const fetchModelInfo = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/sarima/info`);
      if (!res.ok) throw new Error("Failed to fetch model info");
      const data = await res.json();
      setModelInfo(data);
    } catch (err) {
      console.error("Model info error:", err);
    }
  };

  const fetchForecast = async (forecastDays) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/sarima/forecast?days=${forecastDays}`);
      if (!res.ok) throw new Error("Failed to fetch forecast");
      const data = await res.json();
      setForecast(data);
      setDays(forecastDays);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDaysChange = (newDays) => {
    fetchForecast(newDays);
  };

  if (error) {
    return (
      <div
        style={{
          padding: "2rem",
          backgroundColor: "#1a1a1a",
          color: "#fff",
          borderRadius: "8px",
          border: "1px solid #ef4444",
          maxWidth: "1200px",
          margin: "2rem auto",
        }}
      >
        <h2 style={{ color: "#ef4444", marginBottom: "1rem" }}>⚠️ Error</h2>
        <p style={{ marginBottom: "1rem" }}>{error}</p>
        <p style={{ fontSize: "0.9rem", color: "#999" }}>
          Make sure the backend is running and the SARIMA model is trained.
          Run: <code style={{ backgroundColor: "#222", padding: "0.2rem 0.5rem" }}>python train_model.py</code>
        </p>
      </div>
    );
  }

  if (loading) {
    return (
      <div
        style={{
          padding: "2rem",
          textAlign: "center",
          color: "#ccc",
        }}
      >
        <div style={{ fontSize: "2rem", marginBottom: "1rem" }}>⏳</div>
        <p>Loading SARIMA forecast...</p>
      </div>
    );
  }

  if (!forecast) return null;

  const chartData = forecast.predictions.map((p) => ({
    day: p.day,
    date: p.date,
    forecast: parseFloat(p.forecast.toFixed(2)),
    lower: parseFloat(p.lower_ci.toFixed(2)),
    upper: parseFloat(p.upper_ci.toFixed(2)),
    range: parseFloat((p.upper_ci - p.lower_ci).toFixed(2)),
  }));

  const avgTemp = (chartData.reduce((sum, d) => sum + d.forecast, 0) / chartData.length).toFixed(1);
  const maxTemp = Math.max(...chartData.map(d => d.forecast)).toFixed(1);
  const minTemp = Math.min(...chartData.map(d => d.forecast)).toFixed(1);

  return (
    <div
      style={{
        padding: "2rem",
        backgroundColor: "#121212",
        color: "#fff",
        minHeight: "100vh",
      }}
    >
      <div style={{ maxWidth: "1200px", margin: "0 auto" }}>
        {/* Header Stats */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
            gap: "1.5rem",
            marginBottom: "2rem",
          }}
        >
          <div
            style={{
              backgroundColor: "#222",
              padding: "1.5rem",
              borderRadius: "8px",
              border: "1px solid #333",
            }}
          >
            <p style={{ color: "#999", fontSize: "0.9rem", marginBottom: "0.5rem" }}>
              📍 Location
            </p>
            <p style={{ fontSize: "1.5rem", fontWeight: "bold" }}>
              {forecast.location}
            </p>
          </div>

          <div
            style={{
              backgroundColor: "#222",
              padding: "1.5rem",
              borderRadius: "8px",
              border: "1px solid #333",
            }}
          >
            <p style={{ color: "#999", fontSize: "0.9rem", marginBottom: "0.5rem" }}>
              📊 Avg Temperature
            </p>
            <p style={{ fontSize: "1.5rem", fontWeight: "bold", color: "#3b82f6" }}>
              {avgTemp}°C
            </p>
          </div>

          <div
            style={{
              backgroundColor: "#222",
              padding: "1.5rem",
              borderRadius: "8px",
              border: "1px solid #333",
            }}
          >
            <p style={{ color: "#999", fontSize: "0.9rem", marginBottom: "0.5rem" }}>
              📈 Max / Min
            </p>
            <p style={{ fontSize: "1.5rem", fontWeight: "bold" }}>
              <span style={{ color: "#ef4444" }}>{maxTemp}°C</span>
              {" / "}
              <span style={{ color: "#3b82f6" }}>{minTemp}°C</span>
            </p>
          </div>

          <div
            style={{
              backgroundColor: "#222",
              padding: "1.5rem",
              borderRadius: "8px",
              border: "1px solid #333",
            }}
          >
            <p style={{ color: "#999", fontSize: "0.9rem", marginBottom: "0.5rem" }}>
              📅 Forecast Days
            </p>
            <p style={{ fontSize: "1.5rem", fontWeight: "bold", color: "#10b981" }}>
              {forecast.forecast_days} days
            </p>
          </div>
        </div>

        {/* Model Info */}
        {modelInfo && (
          <div
            style={{
              backgroundColor: "#222",
              padding: "1.5rem",
              borderRadius: "8px",
              marginBottom: "2rem",
              border: "1px solid #333",
            }}
          >
            <h3 style={{ marginBottom: "1rem" }}>🔧 Model Information</h3>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
                gap: "1rem",
                fontSize: "0.9rem",
              }}
            >
              <div>
                <p style={{ color: "#999" }}>Model Type</p>
                <p style={{ fontWeight: "bold", color: "#3b82f6" }}>
                  {modelInfo.model}
                </p>
              </div>
              <div>
                <p style={{ color: "#999" }}>Order (p,d,q)</p>
                <p style={{ fontWeight: "bold", color: "#3b82f6" }}>
                  {modelInfo.order}
                </p>
              </div>
              <div>
                <p style={{ color: "#999" }}>Seasonal Order (P,D,Q,m)</p>
                <p style={{ fontWeight: "bold", color: "#3b82f6" }}>
                  {modelInfo.seasonal_order}
                </p>
              </div>
              <div>
                <p style={{ color: "#999" }}>Training Data Range</p>
                <p style={{ fontWeight: "bold", color: "#3b82f6" }}>
                  {modelInfo.data_range}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Days Selector */}
        <div
          style={{
            display: "flex",
            gap: "1rem",
            alignItems: "center",
            marginBottom: "2rem",
            flexWrap: "wrap",
          }}
        >
          <label style={{ fontSize: "1rem", fontWeight: "600" }}>
            📅 Select Forecast Period:
          </label>
          <div style={{ display: "flex", gap: "0.5rem" }}>
            {[7, 14, 30].map((day) => (
              <button
                key={day}
                onClick={() => handleDaysChange(day)}
                style={{
                  padding: "0.75rem 1.5rem",
                  borderRadius: "6px",
                  border: "1px solid #333",
                  backgroundColor: days === day ? "#3b82f6" : "#222",
                  color: days === day ? "#fff" : "#ccc",
                  cursor: "pointer",
                  fontWeight: days === day ? "bold" : "normal",
                  transition: "all 0.3s",
                }}
              >
                {day} days
              </button>
            ))}
          </div>
        </div>

        {/* Tabs */}
        <div style={{ marginBottom: "2rem" }}>
          <div style={{ display: "flex", gap: "1rem", marginBottom: "1rem" }}>
            <button
              onClick={() => setActiveTab("chart")}
              style={{
                padding: "0.75rem 1.5rem",
                borderRadius: "6px",
                border: "none",
                backgroundColor: activeTab === "chart" ? "#3b82f6" : "#222",
                color: "#fff",
                cursor: "pointer",
                fontWeight: activeTab === "chart" ? "bold" : "normal",
              }}
            >
              📊 Chart View
            </button>
            <button
              onClick={() => setActiveTab("table")}
              style={{
                padding: "0.75rem 1.5rem",
                borderRadius: "6px",
                border: "none",
                backgroundColor: activeTab === "table" ? "#3b82f6" : "#222",
                color: "#fff",
                cursor: "pointer",
                fontWeight: activeTab === "table" ? "bold" : "normal",
              }}
            >
              📋 Table View
            </button>
          </div>

          {/* Chart Tab */}
          {activeTab === "chart" && (
            <div
              style={{
                backgroundColor: "#222",
                padding: "1.5rem",
                borderRadius: "8px",
                border: "1px solid #333",
              }}
            >
              <h3 style={{ marginBottom: "1rem" }}>Temperature Forecast</h3>
              <ResponsiveContainer width="100%" height={400}>
                <ComposedChart data={chartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid stroke="#444" />
                  <XAxis
                    dataKey="date"
                    tick={{ fontSize: 12 }}
                    interval={Math.floor(chartData.length / 8) || 0}
                  />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#333",
                      border: "1px solid #444",
                      borderRadius: "6px",
                    }}
                    formatter={(value) => value.toFixed(2) + "°C"}
                    labelFormatter={(label) => `${label}`}
                  />
                  <Bar
                    dataKey="range"
                    fill="rgba(59, 130, 246, 0.2)"
                    name="95% CI Range"
                    yAxisId="right"
                  />
                  <Line
                    type="monotone"
                    dataKey="forecast"
                    stroke="#3b82f6"
                    name="Forecast"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="upper"
                    stroke="#ef4444"
                    strokeDasharray="5 5"
                    name="Upper CI"
                    strokeWidth={1}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="lower"
                    stroke="#ef4444"
                    strokeDasharray="5 5"
                    name="Lower CI"
                    strokeWidth={1}
                    dot={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Table Tab */}
          {activeTab === "table" && (
            <div
              style={{
                backgroundColor: "#222",
                padding: "1.5rem",
                borderRadius: "8px",
                border: "1px solid #333",
                overflowX: "auto",
              }}
            >
              <h3 style={{ marginBottom: "1rem" }}>Detailed Forecast Table</h3>
              <table
                style={{
                  width: "100%",
                  borderCollapse: "collapse",
                  fontSize: "0.9rem",
                }}
              >
                <thead>
                  <tr style={{ backgroundColor: "#333", borderBottom: "2px solid #444" }}>
                    <th style={{ padding: "12px", textAlign: "left" }}>Day</th>
                    <th style={{ padding: "12px", textAlign: "left" }}>Date</th>
                    <th style={{ padding: "12px", textAlign: "center" }}>Forecast (°C)</th>
                    <th style={{ padding: "12px", textAlign: "center" }}>Lower CI</th>
                    <th style={{ padding: "12px", textAlign: "center" }}>Upper CI</th>
                    <th style={{ padding: "12px", textAlign: "center" }}>Range</th>
                  </tr>
                </thead>
                <tbody>
                  {forecast.predictions.map((pred, idx) => (
                    <tr
                      key={idx}
                      style={{
                        borderBottom: "1px solid #333",
                        backgroundColor: idx % 2 === 0 ? "#1a1a1a" : "#222",
                      }}
                    >
                      <td style={{ padding: "10px" }}>{pred.day}</td>
                      <td style={{ padding: "10px" }}>{pred.date}</td>
                      <td
                        style={{
                          padding: "10px",
                          textAlign: "center",
                          fontWeight: "bold",
                          color: "#3b82f6",
                        }}
                      >
                        {pred.forecast.toFixed(1)}
                      </td>
                      <td style={{ padding: "10px", textAlign: "center", color: "#ef4444" }}>
                        {pred.lower_ci.toFixed(1)}
                      </td>
                      <td style={{ padding: "10px", textAlign: "center", color: "#ef4444" }}>
                        {pred.upper_ci.toFixed(1)}
                      </td>
                      <td
                        style={{
                          padding: "10px",
                          textAlign: "center",
                          color: "#10b981",
                        }}
                      >
                        {(pred.upper_ci - pred.lower_ci).toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Footer */}
        <p
          style={{
            marginTop: "2rem",
            fontSize: "0.8rem",
            color: "#666",
            textAlign: "center",
            borderTop: "1px solid #333",
            paddingTop: "1rem",
          }}
        >
          🔮 SARIMA Weather Forecast • Trained on Tokyo historical data (2022-2024) • 95% Confidence Intervals
        </p>
      </div>
    </div>
  );
}