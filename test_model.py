import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ---------------- LOAD DATA ---------------- #

df = pd.read_csv("Dataset.csv")

# ---------------- HANDLE '?' VALUES ---------------- #

# Convert '?' to NaN
df.replace('?', np.nan, inplace=True)

# Convert numeric columns properly
num_cols = [
    'yr','mnth','hr','temp','atemp','hum','windspeed',
    'casual','registered','cnt'
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values (median best)
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# ---------------- DROP NON-USEFUL COLUMNS ---------------- #

df = df.drop(['instant', 'dteday'], axis=1)

# ---------------- ONE-HOT ENCODING ---------------- #

df = pd.get_dummies(
    df,
    columns=['season','weathersit','holiday','workingday','weekday'],
    drop_first=True
)

# ---------------- TRAIN-TEST SPLIT ---------------- #

X = df.drop('cnt', axis=1)
y = df['cnt']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL TRAINING ---------------- #

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

print("✅ Model trained successfully")

# ---------------- SAVE MODEL ---------------- #

joblib.dump(model, "bike_demand_model.pkl")
print("✅ bike_demand_model.pkl saved successfully")
