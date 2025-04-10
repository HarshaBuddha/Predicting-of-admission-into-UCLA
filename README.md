# Predicting Admission into UCLA

This repository contains a complete web application that predicts the likelihood of admission into UCLA based on various applicant features. The project includes a **frontend (React.js)**, **backend (Express.js)**, and **Flask API** for making predictions using a trained machine learning model.

## Table of Contents

- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Contributing](#contributing)

## Project Overview

The UCLA Admission Predictor helps students estimate their chances of admission based on input features such as GRE scores, TOEFL scores, GPA, and research experience. The model was trained using machine learning techniques and deployed via a REST API.

## Folder Structure
```
Prediction-of-admission-into-UCLA/
│── README.md                          # Project overview
│── app/
│   ├── ucla-admission-frontend/       # React frontend
│   │   ├── README.md                  # Instructions for frontend setup
│   ├── express-backend/               # Express.js backend
│   │   ├── README.md                  # Instructions for backend setup
│   ├── Flask/                         # Flask API
│   │   ├── README.md                  # Instructions for Flask API setup
│── admission_predict.csv              # (Optional) Dataset used for training
│── Predicting_Admission_Complete.ipynb     #Complete File
```

## Installation and Setup
Each part of the project has its own setup instructions. Navigate to the respective folders and follow the installation guides in their README files:

- **Frontend (React.js):** `cd app/` → Follow `README.md`
- **Backend (Express.js):** `cd app/` → Follow `README.md`
- **Flask API (Machine Learning Model):** `cd app/` → Follow `README.md`

## Usage
Once all services are running:
- Open the frontend in a browser.
- Enter applicant details.
- Click **Predict** to get an admission probability.

## Model Details
The prediction model was trained using:
- **Features:** GRE, TOEFL, University Rating, SOP, LOR, CGPA, Research
- **Final Model:** Linear Regression (best accuracy after hyperparameter tuning)
- **Evaluation Metrics:** MSE, R2 Score

## Results
The model provides a probability score (0-100%) indicating the likelihood of admission.

## Contributing
Contributions are welcome! Open an issue or submit a pull request if you want to improve the project.

