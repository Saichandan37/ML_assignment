import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import custom model functions
from logistic_regression import train_logistic
from decision_tree import train_dt
from knn_model import train_knn
from naive_bayes import train_nb
from random_forest import train_rf
from xgboost_model import train_xgboost

def run_pipeline():
    # Step 1: Download/Load Dataset Online
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # Step 2: Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Step 3: Run Models
    results = {}
    results['Logistic Regression'] = train_logistic(X_train, X_test, y_train, y_test)
    results['Decision Tree'] = train_dt(X_train, X_test, y_train, y_test)
    results['kNN'] = train_knn(X_train, X_test, y_train, y_test)
    results['Naive Bayes'] = train_nb(X_train, X_test, y_train, y_test)
    results['Random Forest'] = train_rf(X_train, X_test, y_train, y_test)
    results['XGBoost'] = train_xgboost(X_train, X_test, y_train, y_test)

    # Step 4: Generate Comparison Table
    report_df = pd.DataFrame(results).T
    print("\n" + "="*50)
    print("COPY THIS TABLE INTO YOUR README.MD")
    print("="*50)
    print(report_df.to_markdown())
    print("="*50)

if __name__ == "__main__":
    run_pipeline()