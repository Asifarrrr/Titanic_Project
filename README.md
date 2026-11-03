# 🚢 Titanic Survival Intelligence Engine

A professional full-stack machine learning web application that predicts passenger survival probabilities on the Titanic 🧊. It integrates a trained Scikit-Learn classification model with a **FastAPI** backend server that simultaneously serves an interactive HTML frontend dashboard and handles prediction API requests ⚡.

---

## 🎯 Project Overview

The goal of this project is to develop a robust predictive pipeline capable of estimating passenger survival probabilities using key features such as passenger class, age, gender, fare, and family relationships 📊. 

Unlike traditional static scripts, this project bridges machine learning and full-stack engineering by serving a live dashboard straight from your local backend server, making it fully ready for GitHub deployment and cloud scaling ☁️.

---

## 📂 Project Architecture

```text
titanic-project/
├── app/
│   ├── main.py              # FastAPI server & route handlers
│   └── templates/
│       └── dashboard.html   # Frontend user interface
├── data/                    # Dataset and dataframes (.pkl / .csv)
├── models/                  # Serialized ML model, scaler, and columns
├── screenshots/             # App preview images for documentation
├── requirements.txt         # Project Python dependencies
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone [https://github.com/Asifarrrr/Titanic_Project.git](https://github.com/Asifarrrr/Titanic_Project.git)
cd Titanic_Project
```

### 2. Set Up a Virtual Environment & Install Dependencies
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Backend & Frontend (Unified Server)
Start the Uvicorn server from the root directory to launch the app 🚀:
```bash
uvicorn app.main:app --reload
```
*(Note: The server will automatically trigger your browser to open the dashboard page on startup!)*

### 4. Access the Application
* **Interactive Dashboard:** Open your browser and navigate to **`http://127.0.0.1:8000`** 🌐
* **API Documentation (Swagger UI):** Visit **`http://127.0.0.1:8000/docs`** 📄

---

## 🛠️ Tech Stack & Workflow

- **Python** – Core programming language 🐍.
- **FastAPI & Uvicorn** – Backend web server and REST API framework ⚡.
- **Pandas, NumPy & Scikit-learn** – Data preprocessing, scaling, and classification modeling 📈.
- **Joblib** – Model serialization and persistence (`.pkl` files) 💾.
- **HTML5, CSS3 & JavaScript (Fetch API)** – Interactive frontend web dashboard 🖥️.

---

## 📌 Project Objectives

- Master the end-to-end Machine Learning pipeline from data cleaning to model deployment 🛠️.
- Transition from ad-hoc scripts to structured, production-ready full-stack architecture 🏗️.
- Learn how to serve frontend interfaces directly through a FastAPI Python backend 🔗.
- Maintain clean version control workflows using Git and GitHub 🐙.

---

**This project was developed as part of my journey in Data Science and Machine Learning, combining predictive modeling, API development, and web UI integration into one practical portfolio piece 🌟.**
