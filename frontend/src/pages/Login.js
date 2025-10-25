// src/pages/Login.js
import React, { useState } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom'; // Removed 'useNavigate'

function Login({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  // const navigate = useNavigate(); // This line was unused and removed

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    try {
      // Send login request to the Flask server
      const response = await axios.post('http://localhost:5000/login', {
        username,
        password,
      });
      // Call the onLogin function from App.js
      onLogin(response.data.username);
    } catch (err) {
      setError(err.response?.data?.message || 'Login failed');
    }
  };

  return (
    <div className="auth-container">
      <form onSubmit={handleSubmit} className="auth-form">
        <h2>Login</h2>
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <button type="submit">Login</button>
        {error && <p className="auth-error">{error}</p>}
      </form>
      <p>
        Don't have an account? <Link to="/register">Register here</Link>
      </p>
    </div>
  );
}

export default Login;