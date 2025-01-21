from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import joblib

app = FastAPI()

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

joblib.dump(model, '../breast_cancer_log_reg_model.pkl')
joblib.dump(scaler, '../scaler.pkl')

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)


class PredictionRequest(BaseModel):
    features: list


@app.on_event("startup")
def load_model_and_scaler():
    global model, scaler
    model = joblib.load('../breast_cancer_log_reg_model.pkl')
    scaler = joblib.load('../scaler.pkl')


@app.get("/")
def root():
    return {"message": "Welcome to the Breast Cancer Prediction API!"}


@app.post("/predict/")
def predict(request: PredictionRequest):
    try:

        if len(request.features) != 30:
            raise HTTPException(status_code=400, detail="Expected 30 features for prediction.")

        input_features = np.array(request.features).reshape(1, -1)
        input_scaled = scaler.transform(input_features)

        prediction = model.predict(input_scaled)
        result = "benign" if prediction[0] == 1 else "malignant"

        return {
            "prediction": result,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
