import pandas as pd
import numpy as np
import joblib
import os
import opendatasets as od
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
print("Step 1: Loading Data...")

csv_path = './e-commerce-analytics-swiggy-zomato-blinkit/Ecommerce_Delivery_Analytics_New.csv'

# Download if missing
if not os.path.exists(csv_path) and not os.path.exists('Ecommerce_Delivery_Analytics_New.csv'):
    try:
        od.download("https://www.kaggle.com/datasets/logiccraftbyhimanshi/e-commerce-analytics-swiggy-zomato-blinkit")
    except:
        pass

# Load File
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
elif os.path.exists('Ecommerce_Delivery_Analytics_New.csv'):
    df = pd.read_csv('Ecommerce_Delivery_Analytics_New.csv')
else:
    print("❌ Error: CSV file not found. Please download it from Kaggle.")
    exit()

# ---------------------------------------------------------
# 2. Smart Feature Engineering & Cleaning
# ---------------------------------------------------------
print("Step 2: Cleaning & Creating Smart Features...")

# A. Fix Platform Names (Typos)
df['Platform'] = df['Platform'].astype(str).str.strip().str.title()
typo_corrections = {'Jio Mart': 'JioMart', 'Jiomart': 'JioMart', 'Swiggy': 'Swiggy Instamart', 
                    'Blink It': 'Blinkit', 'Blinikit': 'Blinkit', 'Amazon': 'Amazon Fresh'}
df['Platform'] = df['Platform'].replace(typo_corrections)
valid_platforms = ['Blinkit', 'JioMart', 'Swiggy Instamart', 'Zepto', 'Amazon Fresh', 'BigBasket']
df = df[df['Platform'].isin(valid_platforms)].copy()

# B. Fix Time Outliers (IQR Method)
Q1 = df['Delivery Time (Minutes)'].quantile(0.25)
Q3 = df['Delivery Time (Minutes)'].quantile(0.75)
upper_bound = Q3 + 1.5 * (Q3 - Q1)
df = df[df['Delivery Time (Minutes)'] <= upper_bound]

# C. Extract Hour
try:
    df['Order_Hour'] = pd.to_datetime(df['Order Date & Time'], errors='coerce').dt.hour
    df['Order_Hour'] = df['Order_Hour'].fillna(12).astype(int)
except:
    df['Order_Hour'] = 12

# D. Auto-Detect Worst Traffic Hour
hourly_delays = df[df['Delivery Delay'] == 'Yes'].groupby('Order_Hour').size()
total_orders = df.groupby('Order_Hour').size()
worst_hour = (hourly_delays / total_orders).fillna(0).idxmax()
print(f"   -> 🚨 Insight: Worst Traffic is around Hour {worst_hour}:00")

# E. Create Logic Features (Must match App)
df['Is_Rush_Hour'] = df['Order_Hour'].apply(lambda x: 1 if (worst_hour - 2) <= x <= (worst_hour + 2) else 0)
df['Is_High_Value'] = df['Order Value (INR)'].apply(lambda x: 1 if x > 800 else 0)

# F. Define Target
target_col = 'Delivery Delay'
drop_cols = ['Order ID', 'Customer ID', 'Order Date & Time', 'Customer Feedback', 
             'Refund Requested', 'Delivery Time (Minutes)', 'Delivery Delay', 'Service Rating']

# ---------------------------------------------------------
# 3. Strict 50/50 Balancing
# ---------------------------------------------------------
print("Step 3: Balancing Data (50% Delayed / 50% On Time)...")
df_delay = df[df[target_col] == 'Yes']
df_ontime = df[df[target_col] == 'No']

n_samples = min(len(df_delay), len(df_ontime))
df_balanced = pd.concat([
    df_delay.sample(n=n_samples, random_state=42),
    df_ontime.sample(n=n_samples, random_state=42)
])

X = df_balanced.drop(columns=drop_cols, errors='ignore')
y = df_balanced[target_col]

# Encode Target (Yes=1, No=0)
le = LabelEncoder()
y = le.fit_transform(y)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ---------------------------------------------------------
# 4. Hyperparameter Tuning (Aggressive Model)
# ---------------------------------------------------------
print("Step 4: Training & Tuning Gradient Boosting Model...")

# Preprocessing Pipeline
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Aggressive Grid
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.1, 0.2],
    'classifier__max_depth': [5, 10]
}

# Run Search
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=1, verbose=1)
grid_search.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. Save Best Model
# ---------------------------------------------------------
best_model = grid_search.best_estimator_
print(f"\n✅ Training Complete! Test Accuracy: {best_model.score(X_test, y_test):.2%}")

joblib.dump(best_model, 'my_modelss.pkl')
print("✅ Model saved as 'my_model.pkl'")