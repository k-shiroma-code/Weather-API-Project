import React from "react";

export default function Header() {
  return (
    <header style={{ backgroundColor: "#1f1f1f", padding: "1rem" }}>
      <nav style={{ display: "flex", justifyContent: "space-between", maxWidth: "1200px", margin: "0 auto" }}>
        <a href="/" style={{ color: "#0ea5e9", fontWeight: "bold", fontSize: "1.5rem" }}>Weather Dashboard</a>
        <div style={{ display: "flex", gap: "1rem" }}>
          <a href="/" style={{ color: "#eee" }}>Home</a>
          <a href="/forecast" style={{ color: "#eee" }}>Forecast</a>
          <a href="/table_charts" style={{ color: "#eee" }}>Data Table</a>
        </div>
      </nav>
    </header>
  );
}
