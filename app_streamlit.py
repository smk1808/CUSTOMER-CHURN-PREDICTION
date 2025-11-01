import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# -------------------------------
# Load trained model
# -------------------------------
try:
    model = joblib.load('D:\CustomerChurnDashboard\models\churn_model.pkl')
except FileNotFoundError:
    st.error("❌ Model file not found. Please ensure 'churn_model.pkl' is inside the 'models/' folder.")
    st.stop()

st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")

st.markdown("""
This dashboard lets you:
1. Upload customer data for churn prediction  
2. Predict churn for individual customers  
3. Visualize feature importance for business insights
---
""")

# -------------------------------
# Function: Clean uploaded data
# -------------------------------
def clean_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean user-uploaded customer data for prediction."""
    df = df.copy()

    # Replace blank strings or spaces with NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # Fill missing numeric values with median, categorical with mode
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)

    # Strip whitespace from string columns
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    return df


# -------------------------------
# Section 1: Upload data for batch prediction
# -------------------------------
st.sidebar.header("Upload customer data (optional)")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with customer data", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = clean_input_dataframe(data)

    st.subheader("📂 Uploaded Data Preview")
    st.dataframe(data.head())

    if st.button("🚀 Predict for Uploaded Data"):
        try:
            preds = model.predict(data)
            probs = model.predict_proba(data)[:, 1]
            result_df = data.copy()
            result_df['Churn_Probability'] = probs.round(2)
            result_df['Prediction'] = np.where(preds == 1, 'Churn', 'Stay')

            st.success("✅ Predictions generated successfully!")
            st.dataframe(result_df.head())

            csv_download = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_download,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Prediction failed. Error: {e}")

# -------------------------------
# Section 2: Single customer prediction
# -------------------------------
st.header("🔮 Predict for a Single Customer")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    tenure = st.number_input("Tenure (months)", min_value=0)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col3:
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0)

input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [SeniorCitizen],
    'Partner': [Partner],
    'Dependents': [Dependents],
    'tenure': [tenure],
    'PhoneService': [PhoneService],
    'MultipleLines': [MultipleLines],
    'InternetService': [InternetService],
    'OnlineSecurity': [OnlineSecurity],
    'OnlineBackup': [OnlineBackup],
    'DeviceProtection': [DeviceProtection],
    'TechSupport': [TechSupport],
    'StreamingTV': [StreamingTV],
    'StreamingMovies': [StreamingMovies],
    'Contract': [Contract],
    'PaperlessBilling': [PaperlessBilling],
    'PaymentMethod': [PaymentMethod],
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges]
})

if st.button("Predict Churn for Customer"):
    try:
        input_data = clean_input_dataframe(input_data)
        proba = model.predict_proba(input_data)[:, 1][0]
        result = "🔴 Likely to Churn" if proba >= 0.5 else "🟢 Likely to Stay"
        st.metric(label="Prediction Result", value=result, delta=f"{proba:.2f} probability")
    except Exception as e:
        st.error(f"Prediction failed. Please check input values. Error: {e}")

# -------------------------------
# Section 3: Feature importance
# -------------------------------
st.header("📈 Feature Importance")

try:
    rf_model = model.named_steps['model']
    ohe = model.named_steps['prep'].named_transformers_['cat'].named_steps['encoder']
    cat_cols = ohe.get_feature_names_out()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    feature_names = np.concatenate([num_cols, cat_cols])
    importances = rf_model.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(15)

    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 15 Features Influencing Churn',
        color='Importance',
        color_continuous_scale='Tealgrn'
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception:
    st.info("Feature importance visualization is available only for tree-based models (e.g., Random Forest, XGBoost).")
