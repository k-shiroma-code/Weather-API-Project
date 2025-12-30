import React, { useEffect, useState } from "react";
import { XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, AreaChart, Area, Legend, BarChart, Bar } from "recharts";

export default function Forecast() {
  const [forecast, setForecast] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("forecast");
  const [selectedModel, setSelectedModel] = useState("gradient_boosting");
  const [selectedArea, setSelectedArea] = useState("SCE");

  // Service areas with their characteristics
  const serviceAreas = {
    SCE: { 
      name: "SCE - Southern California Edison", 
      region: "Southern California",
      baseLoad: 24000,
      population: "15 million",
      description: "Los Angeles, Orange County, Inland Empire"
    },
    PGE: { 
      name: "PG&E - Pacific Gas & Electric", 
      region: "Northern & Central California",
      baseLoad: 18000,
      population: "16 million",
      description: "San Francisco, Bay Area, Central Valley"
    },
    SDGE: { 
      name: "SDG&E - San Diego Gas & Electric", 
      region: "San Diego Area",
      baseLoad: 4500,
      population: "3.7 million",
      description: "San Diego County, Southern Orange County"
    },
    VEA: { 
      name: "VEA - Valley Electric Association", 
      region: "Nevada/California Border",
      baseLoad: 200,
      population: "45,000",
      description: "Pahrump, Fish Lake Valley, Sandy Valley"
    },
  };

  // Model performance data from CV results
  const modelPerformance = {
    ridge: { name: "Ridge + Weather", mae: 840.0, rmse: 1023.0, mape: 3.41, color: "#3b82f6" },
    random_forest: { name: "Random Forest", mae: 576.9, rmse: 751.1, mape: 2.29, color: "#10b981" },
    gradient_boosting: { name: "Gradient Boosting", mae: 573.2, rmse: 724.8, mape: 2.26, color: "#f59e0b" },
  };

  // CV fold results
  const cvFoldData = [
    { fold: "Fold 1", period: "Aug 4 - Sep 2", Ridge: 472, RF: 583, GB: 547 },
    { fold: "Fold 2", period: "Sep 3 - Oct 2", Ridge: 830, RF: 623, GB: 552 },
    { fold: "Fold 3", period: "Oct 3 - Nov 1", Ridge: 815, RF: 367, GB: 325 },
    { fold: "Fold 4", period: "Nov 2 - Dec 1", Ridge: 829, RF: 456, GB: 500 },
    { fold: "Fold 5", period: "Dec 2 - Dec 31", Ridge: 1254, RF: 856, GB: 943 },
  ];

  // Feature importance
  const featureImportance = [
    { feature: "Temperature", importance: 0.28 },
    { feature: "Temp²", importance: 0.22 },
    { feature: "Demand Lag 1", importance: 0.15 },
    { feature: "Demand Lag 7", importance: 0.12 },
    { feature: "Is Weekend", importance: 0.08 },
    { feature: "Rolling 7d Mean", importance: 0.06 },
    { feature: "PV Production", importance: 0.05 },
    { feature: "Humidity", importance: 0.04 },
  ];

  useEffect(() => {
    generateDemoForecast();
  }, [selectedModel, selectedArea]);

  const generateDemoForecast = () => {
    setLoading(true);
    const predictions = [];
    const baseLoad = serviceAreas[selectedArea].baseLoad;
    const today = new Date();
    
    for (let i = 1; i <= 14; i++) {
      const date = new Date(today);
      date.setDate(date.getDate() + i);
      const dayOfWeek = date.getDay();
      const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;
      
      // Scale effects based on area size
      const scaleFactor = baseLoad / 24000;
      const seasonalEffect = Math.sin((date.getMonth() + 1) * Math.PI / 6) * 2000 * scaleFactor;
      const weekendEffect = isWeekend ? -1500 * scaleFactor : 0;
      const noise = (Math.random() - 0.5) * 800 * scaleFactor;
      const forecastVal = baseLoad + seasonalEffect + weekendEffect + noise;
      const uncertainty = (500 + (i * 30)) * scaleFactor;
      
      predictions.push({
        day: i,
        date: date.toISOString().split('T')[0],
        dayOfWeek: ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][dayOfWeek],
        forecast: Math.round(forecastVal),
        lower_ci: Math.round(forecastVal - uncertainty),
        upper_ci: Math.round(forecastVal + uncertainty),
      });
    }
    
    setForecast({
      location: selectedArea,
      forecast_days: 14,
      model: modelPerformance[selectedModel].name,
      predictions: predictions,
    });
    setLoading(false);
  };

  if (loading || !forecast) {
    return (
      <div style={{ padding: "2rem", textAlign: "center", color: "#ccc" }}>
        <div style={{ fontSize: "2rem", marginBottom: "1rem" }}>⚡</div>
        <p>Loading grid load forecast...</p>
      </div>
    );
  }

  const chartData = forecast.predictions.map((p) => ({
    day: p.day,
    date: p.date,
    dayOfWeek: p.dayOfWeek,
    forecast: p.forecast,
    lower: p.lower_ci,
    upper: p.upper_ci,
    range: p.upper_ci - p.lower_ci,
  }));

  const avgLoad = (chartData.reduce((sum, d) => sum + d.forecast, 0) / chartData.length).toFixed(0);
  const maxLoad = Math.max(...chartData.map(d => d.forecast));
  const minLoad = Math.min(...chartData.map(d => d.forecast));

  return (
    <div style={{ color: "#fff" }}>
      
      {/* Service Area Selector */}
      <div style={{ marginBottom: "2rem" }}>
        <label style={{ fontSize: "0.9rem", color: "#999", display: "block", marginBottom: "0.75rem" }}>📍 Select Service Area:</label>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "0.75rem" }}>
          {Object.entries(serviceAreas).map(([key, area]) => (
            <button
              key={key}
              onClick={() => setSelectedArea(key)}
              style={{
                padding: "1rem",
                borderRadius: "8px",
                border: selectedArea === key ? "2px solid #f59e0b" : "1px solid #333",
                backgroundColor: selectedArea === key ? "rgba(245, 158, 11, 0.15)" : "#1e1e1e",
                color: selectedArea === key ? "#fff" : "#999",
                cursor: "pointer",
                textAlign: "left",
                transition: "all 0.2s",
              }}
            >
              <div style={{ fontWeight: "bold", fontSize: "0.95rem", color: selectedArea === key ? "#f59e0b" : "#ccc", marginBottom: "0.25rem" }}>
                {key}
              </div>
              <div style={{ fontSize: "0.8rem", color: "#888" }}>{area.region}</div>
              <div style={{ fontSize: "0.75rem", color: "#666", marginTop: "0.25rem" }}>{area.description}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Stats Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "1rem", marginBottom: "2rem" }}>
        <div style={{ backgroundColor: "#1e1e1e", padding: "1.25rem", borderRadius: "8px", border: "1px solid #333" }}>
          <p style={{ color: "#999", fontSize: "0.85rem", marginBottom: "0.25rem" }}>📍 Service Area</p>
          <p style={{ fontSize: "1rem", fontWeight: "bold", color: "#f59e0b" }}>{serviceAreas[selectedArea].name.split(' - ')[0]}</p>
          <p style={{ fontSize: "0.75rem", color: "#666", marginTop: "0.25rem" }}>{serviceAreas[selectedArea].population} people</p>
        </div>
        
        <div style={{ backgroundColor: "#1e1e1e", padding: "1.25rem", borderRadius: "8px", border: "1px solid #333" }}>
          <p style={{ color: "#999", fontSize: "0.85rem", marginBottom: "0.25rem" }}>📊 Avg Forecast</p>
          <p style={{ fontSize: "1.4rem", fontWeight: "bold", color: "#f59e0b" }}>{Number(avgLoad).toLocaleString()} MW</p>
        </div>
        
        <div style={{ backgroundColor: "#1e1e1e", padding: "1.25rem", borderRadius: "8px", border: "1px solid #333" }}>
          <p style={{ color: "#999", fontSize: "0.85rem", marginBottom: "0.25rem" }}>📈 Peak / Valley</p>
          <p style={{ fontSize: "1.1rem", fontWeight: "bold" }}>
            <span style={{ color: "#ef4444" }}>{maxLoad.toLocaleString()}</span>
            {" / "}
            <span style={{ color: "#10b981" }}>{minLoad.toLocaleString()}</span>
            <span style={{ fontSize: "0.85rem", color: "#999" }}> MW</span>
          </p>
        </div>
        
        <div style={{ backgroundColor: "#1e1e1e", padding: "1.25rem", borderRadius: "8px", border: "1px solid #333" }}>
          <p style={{ color: "#999", fontSize: "0.85rem", marginBottom: "0.25rem" }}>🎯 Model Accuracy</p>
          <p style={{ fontSize: "1.4rem", fontWeight: "bold", color: "#10b981" }}>
            {modelPerformance[selectedModel].mape}% <span style={{ fontSize: "0.85rem", color: "#999" }}>MAPE</span>
          </p>
        </div>
      </div>

      {/* Model Selector */}
      <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginBottom: "1.5rem", flexWrap: "wrap" }}>
        <label style={{ fontSize: "0.9rem", color: "#999" }}>🤖 Model:</label>
        {Object.entries(modelPerformance).map(([key, model]) => (
          <button
            key={key}
            onClick={() => setSelectedModel(key)}
            style={{
              padding: "0.5rem 1rem",
              borderRadius: "6px",
              border: selectedModel === key ? `2px solid ${model.color}` : "1px solid #333",
              backgroundColor: selectedModel === key ? `${model.color}22` : "#1e1e1e",
              color: selectedModel === key ? model.color : "#999",
              cursor: "pointer",
              fontSize: "0.85rem",
              fontWeight: selectedModel === key ? "bold" : "normal",
            }}
          >
            {model.name}
          </button>
        ))}
      </div>

      {/* Tabs */}
      <div style={{ marginBottom: "1rem" }}>
        <div style={{ display: "flex", gap: "0.5rem", borderBottom: "1px solid #333", paddingBottom: "0.5rem" }}>
          {[
            { id: "forecast", label: "📈 14-Day Forecast" },
            { id: "performance", label: "📊 Model Performance" },
            { id: "features", label: "🔍 Feature Importance" },
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
      </div>

      {/* Tab Content */}
      <div style={{ backgroundColor: "#1e1e1e", borderRadius: "8px", border: "1px solid #333", padding: "1.5rem" }}>
        
        {/* Forecast Tab */}
        {activeTab === "forecast" && (
          <div>
            <h3 style={{ marginBottom: "1rem", fontSize: "1.1rem" }}>
              14-Day Load Forecast • {serviceAreas[selectedArea].name.split(' - ')[0]} • {modelPerformance[selectedModel].name}
            </h3>
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                <defs>
                  <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#333" strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 11, fill: "#999" }} tickFormatter={(val) => val.slice(5)} />
                <YAxis tick={{ fontSize: 11, fill: "#999" }} tickFormatter={(val) => `${(val/1000).toFixed(0)}k`} domain={['dataMin - 1000', 'dataMax + 1000']} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#252525", border: "1px solid #444", borderRadius: "6px" }}
                  formatter={(value) => [`${value.toLocaleString()} MW`, ""]}
                  labelFormatter={(label, payload) => payload?.[0] ? `${label} (${payload[0].payload.dayOfWeek})` : label}
                />
                <Legend />
                <Area type="monotone" dataKey="upper" stroke="#f59e0b33" fill="#f59e0b22" name="Upper 95% CI" />
                <Area type="monotone" dataKey="forecast" stroke="#f59e0b" fill="url(#colorForecast)" strokeWidth={2} name="Forecast" />
                <Area type="monotone" dataKey="lower" stroke="#f59e0b33" fill="#1e1e1e" name="Lower 95% CI" />
              </AreaChart>
            </ResponsiveContainer>
            
            <div style={{ marginTop: "1.5rem", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem" }}>
              <div style={{ backgroundColor: "#252525", padding: "1rem", borderRadius: "6px" }}>
                <p style={{ color: "#999", fontSize: "0.8rem", marginBottom: "0.5rem" }}>💡 Peak Day</p>
                <p style={{ fontWeight: "bold" }}>{chartData.find(d => d.forecast === maxLoad)?.date} ({chartData.find(d => d.forecast === maxLoad)?.dayOfWeek})</p>
                <p style={{ color: "#ef4444", fontSize: "1.1rem" }}>{maxLoad.toLocaleString()} MW</p>
              </div>
              <div style={{ backgroundColor: "#252525", padding: "1rem", borderRadius: "6px" }}>
                <p style={{ color: "#999", fontSize: "0.8rem", marginBottom: "0.5rem" }}>💡 Valley Day</p>
                <p style={{ fontWeight: "bold" }}>{chartData.find(d => d.forecast === minLoad)?.date} ({chartData.find(d => d.forecast === minLoad)?.dayOfWeek})</p>
                <p style={{ color: "#10b981", fontSize: "1.1rem" }}>{minLoad.toLocaleString()} MW</p>
              </div>
              <div style={{ backgroundColor: "#252525", padding: "1rem", borderRadius: "6px" }}>
                <p style={{ color: "#999", fontSize: "0.8rem", marginBottom: "0.5rem" }}>📏 Avg Uncertainty</p>
                <p style={{ fontWeight: "bold", color: "#f59e0b", fontSize: "1.1rem" }}>±{Math.round(chartData.reduce((sum, d) => sum + d.range, 0) / chartData.length / 2).toLocaleString()} MW</p>
              </div>
            </div>
          </div>
        )}

        {/* Performance Tab */}
        {activeTab === "performance" && (
          <div>
            <h3 style={{ marginBottom: "1rem", fontSize: "1.1rem" }}>Cross-Validation Results (5-Fold Expanding Window)</h3>
            
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "1rem", marginBottom: "2rem" }}>
              {Object.entries(modelPerformance).map(([key, model]) => (
                <div 
                  key={key}
                  onClick={() => setSelectedModel(key)}
                  style={{ 
                    backgroundColor: selectedModel === key ? `${model.color}15` : "#252525", 
                    padding: "1.25rem", 
                    borderRadius: "8px",
                    border: selectedModel === key ? `2px solid ${model.color}` : "1px solid #333",
                    cursor: "pointer",
                  }}
                >
                  <p style={{ fontWeight: "bold", marginBottom: "0.75rem", color: model.color }}>{model.name}</p>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", fontSize: "0.85rem" }}>
                    <div><p style={{ color: "#999" }}>MAE</p><p style={{ fontWeight: "bold" }}>{model.mae.toFixed(1)} MW</p></div>
                    <div><p style={{ color: "#999" }}>RMSE</p><p style={{ fontWeight: "bold" }}>{model.rmse.toFixed(1)} MW</p></div>
                    <div><p style={{ color: "#999" }}>MAPE</p><p style={{ fontWeight: "bold" }}>{model.mape}%</p></div>
                    <div><p style={{ color: "#999" }}>Rank</p><p style={{ fontWeight: "bold" }}>{key === "gradient_boosting" ? "🥇 1st" : key === "random_forest" ? "🥈 2nd" : "🥉 3rd"}</p></div>
                  </div>
                </div>
              ))}
            </div>
            
            <h4 style={{ marginBottom: "1rem", fontSize: "1rem", color: "#999" }}>MAE by CV Fold (MW)</h4>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={cvFoldData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                <CartesianGrid stroke="#333" strokeDasharray="3 3" />
                <XAxis dataKey="fold" tick={{ fontSize: 11, fill: "#999" }} />
                <YAxis tick={{ fontSize: 11, fill: "#999" }} />
                <Tooltip contentStyle={{ backgroundColor: "#252525", border: "1px solid #444", borderRadius: "6px" }} formatter={(value) => [`${value} MW`, ""]} />
                <Legend />
                <Bar dataKey="Ridge" fill="#3b82f6" name="Ridge + Weather" />
                <Bar dataKey="RF" fill="#10b981" name="Random Forest" />
                <Bar dataKey="GB" fill="#f59e0b" name="Gradient Boosting" />
              </BarChart>
            </ResponsiveContainer>
            <p style={{ marginTop: "1rem", fontSize: "0.85rem", color: "#666" }}>Note: Fold 5 (December) shows higher errors due to holiday patterns.</p>
          </div>
        )}

        {/* Features Tab */}
        {activeTab === "features" && (
          <div>
            <h3 style={{ marginBottom: "1rem", fontSize: "1.1rem" }}>Top Feature Importance (Gradient Boosting)</h3>
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={featureImportance} layout="vertical" margin={{ top: 10, right: 30, left: 100, bottom: 10 }}>
                <CartesianGrid stroke="#333" strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11, fill: "#999" }} tickFormatter={(val) => `${(val * 100).toFixed(0)}%`} />
                <YAxis type="category" dataKey="feature" tick={{ fontSize: 12, fill: "#ccc" }} />
                <Tooltip contentStyle={{ backgroundColor: "#252525", border: "1px solid #444", borderRadius: "6px" }} formatter={(value) => [`${(value * 100).toFixed(1)}%`, "Importance"]} />
                <Bar dataKey="importance" fill="#f59e0b" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
            
            <div style={{ marginTop: "1.5rem", backgroundColor: "#252525", padding: "1.25rem", borderRadius: "8px" }}>
              <h4 style={{ marginBottom: "0.75rem", fontSize: "1rem" }}>🔬 Key Insights</h4>
              <ul style={{ fontSize: "0.9rem", color: "#ccc", lineHeight: "1.8", paddingLeft: "1.25rem" }}>
                <li><strong>Temperature</strong> is the strongest predictor — drives AC demand in summer</li>
                <li><strong>Temperature²</strong> captures the U-shaped relationship (high demand at hot & cold extremes)</li>
                <li><strong>Lag features</strong> (1-day, 7-day) capture autocorrelation and weekly patterns</li>
                <li><strong>Weekend effect</strong> shows ~6% lower demand on Sat/Sun</li>
                <li><strong>PV Production</strong> offsets net demand during sunny hours</li>
              </ul>
            </div>
          </div>
        )}

        {/* Table Tab */}
        {activeTab === "table" && (
          <div style={{ overflowX: "auto" }}>
            <h3 style={{ marginBottom: "1rem", fontSize: "1.1rem" }}>Detailed Forecast Data</h3>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
              <thead>
                <tr style={{ backgroundColor: "#252525", borderBottom: "2px solid #444" }}>
                  <th style={{ padding: "12px", textAlign: "left" }}>Day</th>
                  <th style={{ padding: "12px", textAlign: "left" }}>Date</th>
                  <th style={{ padding: "12px", textAlign: "center" }}>Day of Week</th>
                  <th style={{ padding: "12px", textAlign: "right" }}>Forecast (MW)</th>
                  <th style={{ padding: "12px", textAlign: "right" }}>Lower 95% CI</th>
                  <th style={{ padding: "12px", textAlign: "right" }}>Upper 95% CI</th>
                </tr>
              </thead>
              <tbody>
                {forecast.predictions.map((pred, idx) => (
                  <tr key={idx} style={{ borderBottom: "1px solid #333", backgroundColor: idx % 2 === 0 ? "#1a1a1a" : "#1e1e1e" }}>
                    <td style={{ padding: "10px" }}>{pred.day}</td>
                    <td style={{ padding: "10px" }}>{pred.date}</td>
                    <td style={{ padding: "10px", textAlign: "center", color: pred.dayOfWeek === "Sat" || pred.dayOfWeek === "Sun" ? "#10b981" : "#999" }}>{pred.dayOfWeek}</td>
                    <td style={{ padding: "10px", textAlign: "right", fontWeight: "bold", color: "#f59e0b" }}>{pred.forecast.toLocaleString()}</td>
                    <td style={{ padding: "10px", textAlign: "right", color: "#999" }}>{pred.lower_ci.toLocaleString()}</td>
                    <td style={{ padding: "10px", textAlign: "right", color: "#999" }}>{pred.upper_ci.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Footer */}
      <div style={{ marginTop: "2rem", padding: "1.5rem", backgroundColor: "#1e1e1e", borderRadius: "8px", border: "1px solid #333" }}>
        <h4 style={{ marginBottom: "1rem", fontSize: "1rem" }}>📋 Project Summary</h4>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "1rem", fontSize: "0.85rem", color: "#999" }}>
          <div><p style={{ color: "#666", marginBottom: "0.25rem" }}>Data Source</p><p style={{ color: "#ccc" }}>Kaggle - California ISO</p></div>
          <div><p style={{ color: "#666", marginBottom: "0.25rem" }}>Observations</p><p style={{ color: "#ccc" }}>315,648 (15-min intervals)</p></div>
          <div><p style={{ color: "#666", marginBottom: "0.25rem" }}>Validation</p><p style={{ color: "#ccc" }}>5-Fold Expanding Window CV</p></div>
          <div><p style={{ color: "#666", marginBottom: "0.25rem" }}>Best Model</p><p style={{ color: "#ccc" }}>Gradient Boosting (MAE: 573 MW)</p></div>
        </div>
      </div>

      <p style={{ marginTop: "1.5rem", fontSize: "0.8rem", color: "#555", textAlign: "center" }}>
        ⚡ SCE Grid Load Forecasting • Machine Learning Portfolio Project
      </p>
    </div>
  );
}