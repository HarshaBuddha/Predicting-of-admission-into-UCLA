const express = require('express');
const axios = require('axios');
const cors = require('cors');
const axiosRetry = require('axios-retry').default;
require('dotenv').config();

const app = express();
const FLASK_API_URL = process.env.FLASK_API_URL;

// Enable retry with exponential backoff (3 retries)
axiosRetry(axios, {
  retries: 3,
  retryDelay: axiosRetry.exponentialDelay,
});

app.use(express.json()); // Parse JSON
app.use(cors()); // Enable CORS

// Sample route
app.get('/', (req, res) => {
  res.json({ message: 'Hello all this is an express app' });
});

// Prediction route
app.post('/predict', async (req, res) => {
  try {
    const userInput = req.body;
    console.log("Sending data to Flask API at:", FLASK_API_URL);
    const response = await axios.post(FLASK_API_URL, userInput,{timeout:50000 });
    res.json(response.data);
  } catch (error) {
    console.error('Error calling Flask API:', error.message);
    if (error.response) {
      console.error('Response data:', error.response.data);
    }
    res.status(500).json({ error: 'Failed to fetch prediction from Flask' });
  }
});

const PORT = process.env.PORT || 5001;
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Express server running on port ${PORT}`);
});
