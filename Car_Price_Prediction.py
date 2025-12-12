import pandas as pd
import torch
from torch import nn, optim
# import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import numpy as np

#1. Load Data
Data = pd.read_csv("used_cars.csv")

#2. Cleaning
Milage = Data['milage'].str.replace('mi.', '').str.replace(',','').astype(float)
print(Milage.head())

Price = Data['price'].str.replace('$', '').str.replace(',','').astype(float)
print(Price.head())

def extract_first_float(text):
    match = re.search(r"(\d+\.\d+)", str(text))
    if match:
        return float(match.group(1))
    return None

Data["engine"] = Data["engine"].apply(extract_first_float)
Data["engine"] = Data["engine"].fillna(Data["engine"].mean())

#3. Select Columns from Training.
# Columns[brand, model, model_year, milage, fuel_type, engine, transmission, ext_col, int_col, accident, clean_title, price]
Categorial_col = ['brand', 'model', 'fuel_type', 'transmission']
Numeric_col = ['model_year', 'milage', 'engine']

X_cat = Data[Categorial_col]
X_num = Data[Numeric_col]
y = Data['price']

#4. One-Hot Encoding for Categorical Columns: A method to convert categorical variables into a binary matrix where each category becomes a new column
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_cat_encoded = encoder.fit_transform(X_cat)

X = np.hstack([X_cat_encoded, X_num])

# 5. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 6. BUILD THE MODEL

model = nn.Linear(X_train.shape[1], 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
