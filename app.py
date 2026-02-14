import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, matthews_corrcoef, 
                             confusion_matrix, classification_report)

st.set_page_config(page_title="ML Model Evaluator", layout="wide")

st.title("📊 ML Classification Model Evaluator")

# Sidebar Data Selection
st.sidebar.header("1. Data Source")
data_source = st.sidebar.radio("Select Data Source:", ("Upload CSV", "Use Sample Dataset (Breast Cancer)"))

df = None
target_col = None

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your test CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        target_col = st.sidebar.selectbox("Select Target Column", df.columns, index=len(df.columns)-1)
else:
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    target_col = 'target'
    st.sidebar.success("Sample Dataset Loaded!")

if df is not None:
    st.write("### Data Preview", df.head())

    # Features and Target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model Selection
    st.sidebar.header("2. Model Selection")
    model_choice = st.sidebar.selectbox("Choose a Model", [
        "Logistic Regression", "Decision Tree", "kNN", 
        "Naive Bayes", "Random Forest", "XGBoost"
    ])

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    }

    model = models[model_choice]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Results Section
    st.subheader(f"Results for {model_choice}")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0

    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("AUC", f"{auc:.3f}")
    col3.metric("Precision", f"{prec:.3f}")
    col4.metric("Recall", f"{rec:.3f}")
    col5.metric("F1 Score", f"{f1:.3f}")
    col6.metric("MCC", f"{mcc:.3f}")

    # Visualizations
    st.write("---")
    c1, c2 = st.columns(2)
    with c1:
        st.write("#### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Purples', ax=ax)
        st.pyplot(fig)
    with c2:
        st.write("#### Classification Report")
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

else:
    st.info("Waiting for data selection...")