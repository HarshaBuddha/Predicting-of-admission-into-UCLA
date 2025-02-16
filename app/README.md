# UCLA Admission Predictor

This repository contains the complete application for predicting admission chances into UCLA. The project consists of three main components:

1. **Frontend** - A React.js application for user interaction.
2. **Express.js Backend** - Handles communication between the frontend and Flask API.
3. **Flask API** - Machine learning model for predicting admission chances.

---

## 1️⃣ Frontend (React.js)

### Overview
The frontend is a React-based web application built with Vite. It provides an interface for users to enter academic details and receive admission predictions.

### Tech Stack
- **React.js (Vite)**
- **Tailwind CSS**
- **Axios** (for API requests)
- **React Toastify** (for notifications)

### Folder Structure
```
frontend/
│── src/
│   ├── components/  # UI components (form, buttons, etc.)
│   ├── pages/       # Main pages
│   ├── App.jsx      # Main entry component
│   ├── main.jsx     # Renders the React app
│── public/          # Static assets
│── package.json     # Dependencies and scripts
│── vite.config.js   # Vite configuration
│── .env             # Environment variables
```

### Setup & Installation
1. Navigate to the frontend directory:
   ```bash
   cd app/frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Create a `.env` file and set the backend API URL:
   ```
   VITE_API_URL=http://localhost:5000
   ```
4. Start the development server:
   ```bash
   npm run dev
   ```
5. Open your browser and visit: `http://localhost:5173`

### Build for Production
To generate a production build:
```bash
npm run build
```

---

## 2️⃣ Express.js Backend

### Overview
The backend acts as a bridge between the frontend and Flask API, handling API requests and responses securely.

### Tech Stack
- **Node.js & Express.js**
- **Axios** (to communicate with Flask API)
- **dotenv** (for environment variables)
- **CORS** (to allow cross-origin requests)

### Folder Structure
```
backend/
├── server.js        # Main Express server   
│── .env             # Environment variables
│── package.json     # Dependencies and scripts
```

### Setup & Installation
1. Navigate to the backend directory:
   ```bash
   cd app/express-backend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Create a `.env` file and set the Flask API URL:
   ```
   FLASK_API_URL=http://localhost:5000
   ```
4. Start the Express server:
   ```bash
   npm start
   ```
5. The server will run on: `http://localhost:5001`

---

## 3️⃣ Flask API (Machine Learning Model)

### Overview
The Flask backend contains the trained machine learning model, which predicts admission chances based on academic data.

### Tech Stack
- **Flask** (for API handling)
- **Scikit-learn** (ML model)
- **NumPy & Pandas** (data processing)
- **Flask-CORS** (to allow cross-origin requests)

### Folder Structure
```
flask_backend/
│── model.pkl       # Trained ML model file
│── scaler.pkl      #Standard Scaler of Trained Model
│── app.py          # Flask application
│── requirements.txt  # Dependencies
```

### Setup & Installation
1. Navigate to the Flask backend directory:
   ```bash
   cd app/Flask
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the Flask API:
   ```bash
   python app.py
   ```
5. The API will be available at: `http://localhost:5000`

---

## 🚀 Running the Full Project
1. **Start Flask API:**
   ```bash
   cd app/Flask && python app.py
   ```
2. **Start Express Backend:(nodemon)**
   ```bash
   cd app/express-backend && npm run dev 
   ```
3. **Start React Frontend:**
   ```bash
   cd app/frontend && npm run dev
   ```

Now, visit `http://localhost:5173` to use the predictor!

---

## 🎯 Features
✅ User-friendly input form
✅ Real-time validation
---


## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.




