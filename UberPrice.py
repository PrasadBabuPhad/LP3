
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Load dataset
df = pd.read_csv("uber.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

# 2. Preprocessing
df.drop(['Unnamed: 0', 'key'], axis=1,
        inplace=True, errors='ignore')

df.drop_duplicates(inplace=True)

df['pickup_datetime'] = pd.to_datetime(
    df['pickup_datetime'], errors='coerce', utc=True
)

df.dropna(inplace=True)

# Extract date and time
df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.day
df['month'] = df['pickup_datetime'].dt.month
df['year'] = df['pickup_datetime'].dt.year
df['weekday'] = df['pickup_datetime'].dt.weekday

df.drop('pickup_datetime', axis=1, inplace=True)

# Remove invalid values
df = df[
    (df['fare_amount'] > 0) &
    (df['passenger_count'].between(1, 6)) &
    (df['pickup_longitude'].between(-75, -72)) &
    (df['dropoff_longitude'].between(-75, -72)) &
    (df['pickup_latitude'].between(40, 42)) &
    (df['dropoff_latitude'].between(40, 42))
]

# 3. Calculate distance using Haversine formula
def distance(row):
    from math import radians, sin, cos, asin, sqrt

    lat1 = radians(row['pickup_latitude'])
    lon1 = radians(row['pickup_longitude'])
    lat2 = radians(row['dropoff_latitude'])
    lon2 = radians(row['dropoff_longitude'])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2

    return 6371 * 2 * asin(sqrt(a))

df['distance'] = df.apply(distance, axis=1)

# 4. Identify and remove outliers using IQR
for col in ['fare_amount', 'distance']:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("\nDataset after preprocessing:")
print(df.shape)

# 5. Visualize outliers
plt.figure(figsize=(8, 4))
sns.boxplot(data=df[['fare_amount', 'distance']])
plt.title("Outlier Detection")
plt.show()

# 6. Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True),
            annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 7. Separate features and target
X = df.drop('fare_amount', axis=1)
y = df['fare_amount']

# 8. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 9. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)

# 10. Random Forest Regression
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

# 11. Evaluate models
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'R2 Score': [
        r2_score(y_test, pred_lr),
        r2_score(y_test, pred_rf)
    ],
    'RMSE': [
        np.sqrt(mean_squared_error(y_test, pred_lr)),
        np.sqrt(mean_squared_error(y_test, pred_rf))
    ],
    'MAE': [
        mean_absolute_error(y_test, pred_lr),
        mean_absolute_error(y_test, pred_rf)
    ]
})

print("\nModel Comparison:")
print(results)

# 12. Compare model performance
results.set_index('Model')[['R2 Score']].plot(
    kind='bar', title='R2 Score Comparison'
)
plt.ylabel("R2 Score")
plt.xticks(rotation=0)
plt.show()

# 13. Actual vs Predicted values
plt.figure(figsize=(8, 5))
plt.scatter(y_test, pred_lr, alpha=0.4, label='Linear Regression')
plt.scatter(y_test, pred_rf, alpha=0.4, label='Random Forest')

plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Actual vs Predicted Fare")
plt.legend()
plt.show()

# 14. Predict fare for a new ride
new_ride = pd.DataFrame([{
    'pickup_longitude': -73.99,
    'pickup_latitude': 40.73,
    'dropoff_longitude': -73.98,
    'dropoff_latitude': 40.75,
    'passenger_count': 2,
    'hour': 14,
    'day': 15,
    'month': 6,
    'year': 2015,
    'weekday': 1,
    'distance': 2.5
}])

prediction = rf.predict(new_ride)

print("\nPredicted Uber Fare: $", round(prediction[0], 2))
