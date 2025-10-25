// src/pages/Dashboard.js
import React from 'react';

function Dashboard() {
  const username = localStorage.getItem('user') || 'User';

  return (
    <div>
      <h2>Welcome to your Dashboard, {username}!</h2>
      <br/>
      <h3>Project Title: Predictive and Empathic AI System for mental health</h3>
      <p>Project use case no: 11</p>
    </div>
  );
}

export default Dashboard;