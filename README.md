# 🏙️ UrbanPulse – Predictive Urban Growth Intelligence Dashboard

## 📌 Overview

**UrbanPulse** is an interactive data-driven dashboard designed to analyze and predict urban growth patterns using machine learning and advanced visual analytics.

The system evaluates factors like infrastructure, pricing trends, and growth velocity to classify regions into actionable growth zones for better decision-making.

---

## 🌐 Live Demo

🚀 **Explore the app:**
https://urban-growth-model-zg3woyssnx5ej3vfeby4th.streamlit.app/

---

## 🎯 Key Features

* 📊 **Growth Zone Classification**

  * 🔥 Hot Zone
  * 🌤 Warm Zone
  * 🌊 Stable Zone
  * ❄️ Cool Zone

* 🤖 **Machine Learning Prediction**

  * Uses **Gradient Boosting Regressor**
  * Captures complex, non-linear patterns in urban data

* 📈 **Interactive Visualizations**

  * Infrastructure vs Growth scatter analysis
  * Zone distribution charts
  * Radar comparison of top neighborhoods
  * Dynamic filtering & exploration

* 🎛 **User Interaction**

  * Real-time filtering
  * Detailed neighborhood insights
  * Clean, responsive UI

---

## 🧠 Tech Stack

* **Python**
* **Streamlit**
* **Pandas / NumPy**
* **Scikit-learn**
* **Plotly**

---

## 🧪 Model Details

* **Algorithm:** Gradient Boosting Regressor

* **Purpose:** Predict urban growth score

* **Why Gradient Boosting?**

  * Handles complex relationships better than linear models
  * Improves prediction accuracy via sequential learning

* **Inputs:**

  * Infrastructure score
  * Price trends
  * Demand indicators

* **Output:**

  * Growth score → used for zone classification

---

## 📊 How It Works

1. Data preprocessing & cleaning
2. Feature engineering (growth velocity, infra score, etc.)
3. Model predicts growth score
4. Zones are classified based on thresholds
5. Interactive dashboard visualizes results

---

## ▶️ Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/urban-growth-model.git
cd urban-growth-model
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
urban-growth-model/
│
├── app.py
├── train.csv
├── requirements.txt
├── README.md
```

---

## 💡 Use Cases

* Real estate investment decisions
* Urban development planning
* Market growth analysis
* Data-driven location insights

---

## 🚀 Future Improvements

* Real-time data integration (APIs)
* Map-based visualization
* Model tuning & comparison
* Cloud scalability

---

## 👩‍💻 Author

**Radhika Jindal**
Data Science & Analytics Enthusiast

---

## ⭐ Final Note

This project demonstrates how machine learning + visualization can transform raw data into meaningful urban growth insights.
