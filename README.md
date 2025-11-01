🧠 **Customer Churn Prediction Dashboard**
Customer Churn Prediction Dashboard — A Streamlit web app that predicts customer churn using a machine learning model trained on the Telco dataset. Includes CSV upload for batch predictions, single-customer churn analysis, and feature importance visualization for business insights.

📊 Overview

This project predicts customer churn (likelihood of customers leaving a service) using machine learning.
It includes:

Data preprocessing, feature engineering, and model training (in Kaggle)

A Streamlit dashboard for live predictions and churn analytics

🚀 Features

✅ Predict churn probability for individual customers
✅ Upload CSVs for bulk predictions
✅ Visualize top features influencing churn
✅ Clean handling of missing or blank values
✅ Ready for deployment on Streamlit Cloud or Heroku

🧩 Tech Stack
Python 3.10+
Pandas / NumPy – Data processing
Scikit-learn – ML model & pipeline
Joblib – Model serialization
Streamlit – Interactive dashboard
Plotly – Visual analytics

📁 Project Structure
CustomerChurnDashboard/
│
├── models/
│   └── churn_model.pkl
├── data/
│   └── (optional) sample_customer_data.csv
├── app_streamlit.py
├── requirements.txt
└── README.md

⚙️ How to Run Locally

Clone the repo

git clone https://github.com/<your-username>/CustomerChurnDashboard.git
cd CustomerChurnDashboard

Create and activate virtual environment

python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Mac/Linux

Install dependencies
pip install -r requirements.txt

Run the app
streamlit run app_streamlit.py

🧠 Model Training
The model was trained using the Telco Customer Churn Dataset on Kaggle:
🔗 https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Feature preprocessing using ColumnTransformer
Model: RandomForestClassifier
Evaluation: ROC-AUC = ~0.83

🌐 Deployment
To deploy on Streamlit Cloud
:
Push your repo to GitHub
Go to Streamlit → “New app”
Connect your repo and select app_streamlit.py
