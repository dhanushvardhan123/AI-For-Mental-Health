// src/App.js
import React, { useState } from 'react';
import { Routes, Route, useNavigate, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import MentalHealthCheckup from './pages/MentalHealthCheckup';
import AboutUs from './pages/AboutUs';
import Login from './pages/Login';
import Register from './pages/Register';
import './App.css'; // We will create this file

function App() {
  // Check if a user is logged in by looking at localStorage
  const [isLoggedIn, setIsLoggedIn] = useState(!!localStorage.getItem('user'));
  const navigate = useNavigate();

  const handleLogin = (username) => {
    localStorage.setItem('user', username); // Store user on login
    setIsLoggedIn(true);
    navigate('/dashboard');
  };

  const handleLogout = () => {
    localStorage.removeItem('user'); // Clear user on logout
    setIsLoggedIn(false);
    navigate('/login');
  };

  return (
    <Routes>
      {isLoggedIn ? (
        // Routes available AFTER login
        <Route path="/" element={<Layout onLogout={handleLogout} />}>
          <Route index element={<Dashboard />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="checkup" element={<MentalHealthCheckup />} />
          <Route path="about" element={<AboutUs />} />
          {/* Any other path redirects to dashboard if logged in */}
          <Route path="*" element={<Navigate to="/dashboard" />} />
        </Route>
      ) : (
        // Routes available BEFORE login
        <>
          <Route path="/login" element={<Login onLogin={handleLogin} />} />
          <Route path="/register" element={<Register />} />
          {/* Any other path redirects to login if not logged in */}
          <Route path="*" element={<Navigate to="/login" />} />
        </>
      )}
    </Routes>
  );
}

export default App;