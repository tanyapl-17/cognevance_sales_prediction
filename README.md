# SalesIQ — AI-Powered Sales Intelligence Platform

SalesIQ is a machine learning portfolio project that analyzes sales performance, estimates revenue, detects unusual sales records, and groups sales transactions into meaningful profiles.

## ✨ Features

- 📊 Interactive sales analytics dashboard
- 💰 Revenue and profit tracking
- 📈 Product and regional performance analysis
- 🤖 Revenue prediction using machine learning
- 🚨 Unusual sales record detection with Isolation Forest
- 🧩 Sales profile segmentation using K-Means clustering
- 📥 Downloadable CSV reports
- 🎨 Interactive visualizations using Plotly and Streamlit

## 🛠️ Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly

## 📁 Project Structure

```text
cognevance_sales_prediction/
├── app.py
├── train_model.py
├── generate_data.py
├── anomaly_detection.py
├── customer_segmentation.py
├── sales_data.csv
├── sales_model.pkl
├── model_comparison.csv
├── model_metrics.json
├── requirements.txt
└── README.md
```

## 🚀 Installation and Setup

1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Generate the sample dataset:

```bash
python generate_data.py
```

5. Train and compare the prediction models:

```bash
python train_model.py
```

6. Launch the dashboard:

```bash
python -m streamlit run app.py
```

## 🧠 Machine Learning

- **Revenue prediction:** compares Ridge Regression, Random Forest, and Gradient Boosting models.
- **Anomaly detection:** uses Isolation Forest to flag unusual sales records.
- **Segmentation:** uses K-Means to group sales records based on revenue, profit, units sold, and discount percentage.

## 📌 Dataset Disclaimer

This project uses synthetically generated sales data for educational and demonstration purposes. Revenue estimates, anomalies, and sales profiles have not been validated for real-world business decisions.

## 👩‍💻 Author

AI/ML and Full-Stack Development Portfolio Project