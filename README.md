Car Price Prediction — Machine Learning Project

This project builds a Machine Learning model that predicts used car prices based on various features such as brand, model, mileage, engine size, transmission, color, accident history, and more.

The dataset includes messy real-world data (e.g., "3.7L V6 Cylinder Engine", "51,000 mi.", "At least 1 accident"), and the project demonstrates data cleaning, feature engineering, encoding, and model training.

Dataset Columns:
    
brandgit 
model
model_year
milage
fuel_type
engine
transmission
ext_col
int_col
accident
clean_title
price

Technologies Used:

 - Python 3
 - Pandas — data cleaning
 - Scikit-Learn (sklearn) — preprocessing & model training
 - Regex (re) — extract numeric values
 - NumPy


How to Run
 
 - Run the command ´python Car_Price_Prediction.py´


Learning from this Project:
 1 - Linear Regression Model

 2 - OneHotEncoder: 

   Learned why machine learning models cannot directly understand categorical/text features.
   Understood how OneHotEncoder converts categories like "Toyota", "Diesel", "Automatic" into numerical binary columns.

 3- ColumnTransformer:

   - Understood the importance of applying different preprocessing steps to different column types.
   - Numeric columns → pass-through
   - Categorical columns → OneHotEncoder.
   - This keeps the preprocessing clean, modular, and consistent.

 4- sklearn:
    
    1- Practiced using Scikit-Learn’s core features:
    2- Train-test split
    3- Preprocessing
    4- Pipeline
    5- Model training
    6- Prediction and evaluation