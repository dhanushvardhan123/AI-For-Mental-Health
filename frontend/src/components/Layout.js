// src/components/Layout.js
import React, { useState } from 'react';
import { Outlet, Link } from 'react-router-dom';

function Layout({ onLogout }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const toggleSidebar = () => setSidebarOpen(!sidebarOpen);
  const closeSidebar = () => setSidebarOpen(false);

  return (
    <div className={`layout ${sidebarOpen ? 'sidebar-open' : ''}`}>
      {/* Top Bar */}
      <header className="top-bar">
        <button onClick={toggleSidebar}>
          ☰ Menu
        </button>
        <h1>Predictive and Empathic AI</h1>
        <button onClick={onLogout}>Logout</button>
      </header>

      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <nav>
          <ul>
            <li><Link to="/dashboard" onClick={closeSidebar}>Dashboard</Link></li>
            <li><Link to="/checkup" onClick={closeSidebar}>Mental Health Checkup</Link></li>
            <li><Link to="/about" onClick={closeSidebar}>About Us</Link></li>
          </ul>
        </nav>
      </aside>

      {/* Main Content Area */}
      <main className="content">
        <Outlet /> {/* This is where your pages (Dashboard, etc.) will be rendered */}
      </main>
    </div>
  );
}

export default Layout;