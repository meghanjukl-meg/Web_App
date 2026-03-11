import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# READ CSV
df = pd.read_csv("beer-servings.csv")

# Dropping missing values in target
df = df.dropna(subset=['total_litres_of_pure_alcohol'])

# GENERATE INFOGRAPHICS (Required for Landing Page)
plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(
    include=[np.number]).corr(), annot=True, cmap='YlGnBu')
plt.title("Correlation Heatmap")
# Save for Flask UI
plt.savefig('static/images/correlation_heatmap.png')
plt.close()

# Pre-processing
categorical_features = ['country', 'continent']
numerical_features = ['beer_servings', 'spirit_servings', 'wine_servings']

# Handling Missing Values
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# SPlit
X = df.drop('total_litres_of_pure_alcohol', axis=1)
y = df['total_litres_of_pure_alcohol']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Model: Linear Regression
lr_model = Pipeline(
    steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
lr_model.fit(X_train, y_train)
lr_r2 = r2_score(y_test, lr_model.predict(X_test))

# Model:Random Forest Regressor
rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                    ('regressor', RandomForestRegressor(random_state=42))])

param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5]
}
# Tuning based on R2-score
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_
rf_r2 = r2_score(y_test, best_rf_model.predict(X_test))


print(f"Linear Regression R2: {lr_r2:.4f}")
print(f"Tuned Random Forest R2: {rf_r2:.4f}")

# Choose best model
if rf_r2 > lr_r2:
    best_model = best_rf_model
    print("Best Model: Random Forest")
else:
    best_model = lr_model
    print("Best Model: Linear Regression")

# pickle.dump(regressor, open('model.pkl', 'wb'))
joblib.dump(best_model, 'best_model.pkl')


e_data = {
    'countries': sorted(df['country'].unique().tolist()),
    'continents': sorted(df['continent'].unique().tolist())
}
joblib.dump(e_data, 'e_data.pkl')
