import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import numpy as np


# 1. Load Dataset

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import re


# 1. LOAD DATA

df = pd.read_csv("used_cars.csv")   # <--- change if needed
print(df.head())


# 2. CLEAN NUMERIC COLUMNS

# clean milage -> remove "mi.", ","
df["milage"] = (
    df["milage"]
    .str.replace(" mi.", "")
    .str.replace(",", "")
    .astype(float)
)

# clean price -> remove "$", ","
df["price"] = (
    df["price"]
    .str.replace("$", "")
    .str.replace(",", "")
    .astype(float)
)

# extract first float from engine text
def extract_first_float(text):
    match = re.search(r"(\d+\.\d+)", str(text))
    if match:
        return float(match.group(1))
    return None

df["engine"] = df["engine"].apply(extract_first_float)
df["engine"] = df["engine"].fillna(df["engine"].mean())


# 3. SELECT COLUMNS FOR TRAINING

categorical_cols = ["brand", "model", "fuel_type", "transmission"]
numeric_cols = ["model_year", "milage", "engine"]

X_cat = df[categorical_cols]
X_num = df[numeric_cols]
y = df["price"]
    
# 4. ONE-HOT ENCODING

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_cat_encoded = encoder.fit_transform(X_cat)

import numpy as np
X = np.hstack([X_cat_encoded, X_num])


# 5. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 6. SIMPLE PYTORCH LINEAR MODEL

model = nn.Linear(X_train.shape[1], 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# 7. TRAIN THE MODEL

epochs = 300
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(X_train)
    loss = loss_fn(predictions, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.2f}")


# 8. TEST

with torch.no_grad():
    test_pred = model(X_test)
    test_loss = loss_fn(test_pred, y_test)

print("\nTest Loss:", test_loss.item())
print("Done!")
print(df.head())


# Accident: convert to Yes/No
df["accident"] = df["accident"].apply(
    lambda x: "Yes" if "accident" in str(x).lower() else "No"
)

# If clean_title has NaN, fill with "Unknown"
df["clean_title"] = df["clean_title"].fillna("Unknown")


# 3. Feature Engineering

df["age"] = 2025 - df["model_year"]


# 4. Select Features

features = [
    "brand", "model", "age", "milage",
    "fuel_type", "engine", "transmission",
    "ext_col", "int_col", "accident", "clean_title"
]

target = "price"

X = df[features]
y = df[target]


# 5. Split Data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Preprocessing

categorical_cols = ["brand", "model", "fuel_type", "transmission",
                    "ext_col", "int_col", "accident", "clean_title"]

numeric_cols = ["age", "milage", "engine"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_cols)
])


# 7. Build Simple Model

model = Pipeline([
    ("prep", preprocessor),
    ("reg", LinearRegression())
])

# 8. Train Model

model.fit(X_train, y_train)


# 9. Evaluate

preds = model.predict(X_test)

print("\nPerformance:")
print("MAE:", round(mean_absolute_error(y_test, preds), 2))
print("R2 :", round(r2_score(y_test, preds), 3))

# 10. Single Prediction Example

sample = {
    "brand": "BMW",
    "model": "4 Series",
    "age": 5,
    "milage": 60000,
    "fuel_type": "Petrol",
    "engine": 4.0,
    "transmission": "Automatic",
    "ext_col": "White",
    "int_col": "Beige",
    "accident": "No",
    "clean_title": "Yes"
}

sample_df = pd.DataFrame([sample])
pred_price = model.predict(sample_df)[0]

print("\nPredicted Price:", round(pred_price, 2))
