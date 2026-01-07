<div align="center">

# 🪨 Rock vs Mine Prediction System

**Advanced Full-Stack ML Web App**  
Predict whether a sonar signal is a **Rock** 🪨 or a **Mine** 💣 with confidence scores and beautiful visualizations.

[![Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-000?style=for-the-badge&logo=vercel&logoColor=white)](https://rockvs-mine.vercel.app)
[![Render](https://img.shields.io/badge/Backend%20on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://rock-vs-mine-backend.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=white)](https://react.dev)

</div>

<p align="center">
  <img src="https://via.placeholder.com/1200x600.png?text=Rock+vs+Mine+Demo+Screenshot" alt="App Screenshot" width="80%"/>
  <br/><br/>
  <em>Modern UI • Real-time Predictions • Interactive Charts • ROC Curve Visualization</em>
</p>

## ✨ Features

- 🎯 **Accurate Predictions** using stacked Logistic Regression + Random Forest
- 📊 **Live Statistics** (Pie, Bar, Trend charts)
- 📈 **ROC Curve** visualization with model details
- 🌙 **Dark Mode** with smooth toggle
- 🎉 **Confetti Celebration** for high-confidence predictions
- ⌨️ **Keyboard Shortcuts** (Ctrl+D → Dark Mode, Ctrl+K → Clear History)
- ⚡ **Fast & Responsive** UI built with Tailwind + Framer Motion
- ☁️ **Fully Deployed** on Vercel (frontend) + Render (backend)

## 🚀 Live Demo

🌐 **Frontend**: [https://rockvs-mine.vercel.app](https://rockvs-mine.vercel.app)  
🔧 **Backend API**: [ https://rockvsmine-3akm.onrender.com]( https://rockvsmine-3akm.onrender.com)  
📚 **API Docs (Swagger)**: [https://rockvsmine-3akm.onrender.com/docs]( https://rockvsmine-3akm.onrender.com/docs)

> Try entering 60 sonar values — use the sample buttons for quick testing!

## 🏗️ Tech Stack

| Category       | Technology                                      |
|----------------|-------------------------------------------------|
| **Frontend**   | React 18, Tailwind CSS, Axios, Framer Motion   |
| **Charts**     | Chart.js + react-chartjs-2                     |
| **Animations** | Framer Motion, canvas-confetti                 |
| **Backend**    | Python 3.12, FastAPI, Uvicorn + Gunicorn       |
| **ML**         | Scikit-learn, NumPy, Pandas, Joblib, Matplotlib|
| **Deployment** | Vercel (Frontend), Render (Backend)            |

## 🧠 Machine Learning Highlights

- **Algorithm**: Stacked ensemble (Logistic Regression on top of Random Forest)
- **Dataset**: Classic Sonar dataset (208 samples, 60 features)
- **Advanced Techniques**:
  - Bidirectional cross-validation
  - Confidence adjustment based on distribution similarity
  - Out-of-distribution (OOD) detection
  - Optional noise injection (training & prediction)
  - Beautiful ROC curve generation (base64)

## ⚡ Quick Start (Local Development)

```bash
# 1. Clone the repo
git clone https://github.com/R357/RockvsMine.git
cd rock-vs-mine-ml-project

# 2. Backend (FastAPI)
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
# → http://localhost:8000

# 3. Frontend (new terminal)
cd ../frontend
npm install
npm start
# → http://localhost:3000
```

## 🛠️ Deployment

### Current Live Deployment

- **Frontend**: Deployed on **Vercel**  
  - Automatic GitHub integration  
  - Instant previews on pull requests  
  - Custom domain support  
  - **URL**: [https://rockvs-mine.vercel.app](https://rockvs-mine.vercel.app)

- **Backend**: Hosted on **Render** (Python service)  
  - Free tier (sleeps after 15 min inactivity)  
  - Auto-deploys on git push  
  - **URL**: [https://rock-vs-mine-backend.onrender.com](https://rock-vs-mine-backend.onrender.com)  
  - **Swagger docs**: [https://rock-vs-mine-backend.onrender.com/docs](https://rock-vs-mine-backend.onrender.com/docs)

### How to Deploy Yourself

1. **Frontend (Vercel)**  
   - Connect your GitHub repo to Vercel  
   - Set **root directory**: `/frontend`  
   - **Build command**: `npm run build`  
   - **Output directory**: `build`  
   - Add environment variable:  


2. **Backend (Render)**  
    - Create **New Web Service** → choose Python  
    - Set **root directory**: `/backend`  
    - **Build command**: `pip install -r requirements.txt`  
    - **Start command**:  

- Free tier works great for demo purposes

## 📌 Future Plans

- 🔒 **User authentication** & persistent prediction history
- 📉 **Model performance dashboard** (accuracy, confusion matrix, calibration plot)
- 📤 **Export prediction history** (CSV/JSON download)
- 🐳 **Docker** + multi-container setup with `docker-compose`
- 🔄 **CI/CD pipeline** using GitHub Actions
- 🌍 **Custom domain** & full HTTPS enforcement
- 📱 **Mobile responsiveness** & Progressive Web App (PWA) support
- 🧪 **A/B testing** for different ML models & hyperparameters
- 📊 **Advanced visualizations** (feature importance, SHAP explanations)


