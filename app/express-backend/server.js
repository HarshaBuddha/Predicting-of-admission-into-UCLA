const express = require('express');
const axios = require('axios');
const cors = require('cors');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT ;
const FLASK_API_URL = process.env.FLASK_API_URL;

app.use(express.json());  // Middleware to parse JSON
app.use(cors());  // Allow frontend requests

//Sample Route
app.get('/',(req,res)=>{
    res.json({'message':'Hello all this is an express app'})
})


// Route to handle prediction requests
app.post('/predict', async (req, res) => {
    try {
        const userInput = req.body;  // Get input from frontend
        const response = await axios.post(FLASK_API_URL, userInput); // Send to Flask API
        res.json(response.data); // Return Flask API response to frontend
    } catch (error) {
        console.error('Error:', error);
        res.status(500).json({ error: 'Failed to fetch prediction' });
    }
});

// Start Express server
app.listen(PORT, () => {
    console.log(`Express server running on http://127.0.0.1:${PORT}`);
});
