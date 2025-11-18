import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ---------------- Tạo dữ liệu ----------------
def generate_water_light_data(n_samples=1000, random_state=42):
    np.random.seed(random_state)
    water = np.random.uniform(0, 100, n_samples)
    light = np.random.uniform(0, 100, n_samples)
    avg = (water + light)/2
    label = (avg >= 80).astype(int)
    X = pd.DataFrame({'water': water, 'light': light})
    y = label
    return train_test_split(X, y, test_size=0.2, random_state=random_state)

X_train, X_test, y_train, y_test = generate_water_light_data()

# ---------------- MLflow Setup ----------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Water_Light_Example")

# ---------------- Train & Tune ----------------
param_list = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 10},
]

best_f1 = 0
best_model = None

for params in param_list:
    with mlflow.start_run():
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Params: {params} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model


# ---------------- Lưu model tốt nhất ra file ----------------

import shutil
import os

if os.path.exists("best_model"):
    shutil.rmtree("best_model")

mlflow.sklearn.save_model(best_model, "best_model")
print("✅ Best model saved to 'best_model' folder")
