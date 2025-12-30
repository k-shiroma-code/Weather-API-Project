import React, { useState } from "react";

export default function Header() {
  const [hoveredLink, setHoveredLink] = useState(null);

  const navLinks = [
    { href: "/", label: "Home", icon: "🏠" },
    { href: "/forecast", label: "Grid Forecast", icon: "⚡" },
    { href: "/table_charts", label: "Weather", icon: "🌍" },
  ];

  return (
    <header
      style={{
        backgroundColor: "#141414",
        padding: "1rem 1.5rem",
        borderBottom: "1px solid #333",
        position: "sticky",
        top: 0,
        zIndex: 100,
        backdropFilter: "blur(10px)",
      }}
    >
      <nav
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          maxWidth: "1400px",
          margin: "0 auto",
        }}
      >
        {/* Logo */}
        <a
          href="/"
          style={{
            color: "#f59e0b",
            fontWeight: "bold",
            fontSize: "1.35rem",
            textDecoration: "none",
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
            transition: "transform 0.2s",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.transform = "scale(1.02)")}
          onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
        >
          <span style={{ fontSize: "1.5rem" }}>⚡</span>
          <span>Energy Dashboard</span>
        </a>

        {/* Nav Links */}
        <div style={{ display: "flex", gap: "0.5rem" }}>
          {navLinks.map((link) => (
            <a
              key={link.href}
              href={link.href}
              onMouseEnter={() => setHoveredLink(link.href)}
              onMouseLeave={() => setHoveredLink(null)}
              style={{
                color: hoveredLink === link.href ? "#f59e0b" : "#ccc",
                textDecoration: "none",
                fontSize: "0.9rem",
                padding: "0.5rem 1rem",
                borderRadius: "6px",
                backgroundColor: hoveredLink === link.href ? "rgba(245, 158, 11, 0.1)" : "transparent",
                transition: "all 0.2s ease",
                display: "flex",
                alignItems: "center",
                gap: "0.4rem",
              }}
            >
              <span style={{ fontSize: "1rem" }}>{link.icon}</span>
              <span>{link.label}</span>
            </a>
          ))}
        </div>
      </nav>
    </header>
  );
}