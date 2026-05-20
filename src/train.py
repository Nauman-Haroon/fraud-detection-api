import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib

# Load data
df = pd.read_csv("data/creditcard.csv")

# Features
X = df.drop("Class", axis=1)

# Target
y = df["Class"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1
)

# Train
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Results
print(classification_report(y_test, preds))

# Save model
joblib.dump(model, "models/model.pkl")

print("Model saved!")