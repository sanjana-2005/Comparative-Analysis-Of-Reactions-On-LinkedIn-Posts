# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor, VotingRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFE
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load dataset with correct encoding
file_path = 'C:/Users/slive/Downloads/MACHINE LEARNING/PROJECT/archive/company_data.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Clean the 'followers' column: remove commas and convert to float
data['followers'] = data['followers'].astype(str).str.replace(',', '').astype(float)

# Check for missing values and impute
data['followers'].fillna(data['followers'].median(), inplace=True)
for col in ["headline", "location", "content", "media_type"]:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Drop columns with the highest percentage of missing values
data.drop(columns=["views", "votes"], inplace=True)

# Handle 'connections' column
if 'connections' in data.columns:
    data['connections'] = data['connections'].astype(str).str.replace(',', '').astype(float)

# Label encode categorical columns
label_encoder = LabelEncoder()
for col in ["headline", "location", "media_type"]:
    data[col] = label_encoder.fit_transform(data[col])

# Handle outliers using IQR method
Q1 = data['reactions'].quantile(0.25)
Q3 = data['reactions'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['reactions'] >= lower_bound) & (data['reactions'] <= upper_bound)]

# Feature Engineering
data['followers_connections_interaction'] = data['followers'] * data['connections']
data['comments_hashtags_interaction'] = data['comments'] * data['num_hashtags']
data['followers_squared'] = data['followers'] ** 2
data['connections_squared'] = data['connections'] ** 2
data['comments_squared'] = data['comments'] ** 2

# Log-transform skewed numerical columns
data['log_followers'] = np.log1p(data['followers'])
data['log_reactions'] = np.log1p(data['reactions'])

# Define features (X) and target (y)
X = data[['headline', 'location', 'log_followers', 'connections', 'media_type', 'num_hashtags', 'comments',
          'followers_connections_interaction', 'comments_hashtags_interaction', 'followers_squared', 
          'connections_squared', 'comments_squared']]
y = data['log_reactions']  # Log-transformed target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize RFE with GradientBoostingRegressor as the estimator
rfe_selector = RFE(estimator=GradientBoostingRegressor(random_state=42), n_features_to_select=8, step=1)

# Fit RFE on training data
rfe_selector.fit(X_train, y_train)

# Get the selected features
selected_features = X.columns[rfe_selector.support_]
print(f"Selected features: {selected_features}")

# Use only the selected features for training and testing
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Model Definitions
xgb_model = XGBRegressor(random_state=42, objective='reg:squarederror')
catboost_model = CatBoostRegressor(random_state=42, verbose=0)
rf_model = RandomForestRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)
lgb_model = lgb.LGBMRegressor(random_state=42)
ada_model = AdaBoostRegressor(random_state=42)

# Preprocessing pipeline (Scaling only numeric features)
preprocessor = ColumnTransformer(
    transformers=[('num', RobustScaler(), selected_features)])

# Hyperparameter Tuning using RandomizedSearchCV for XGBoost
param_dist_xgb = {
    'model__n_estimators': [100, 200, 300, 400],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__max_depth': [3, 5, 7, 9],
    'model__min_child_weight': [1, 3, 5, 7],
    'model__subsample': [0.6, 0.8, 1],
    'model__colsample_bytree': [0.6, 0.8, 1],
    'model__gamma': [0, 0.1, 0.2, 0.3]
}

# XGBoost model pipeline
pipeline_xgb = Pipeline(steps=[('preprocessor', preprocessor), ('model', xgb_model)])

random_search_xgb = RandomizedSearchCV(pipeline_xgb, param_dist_xgb, n_iter=200, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
random_search_xgb.fit(X_train_selected, y_train)

# Best model from XGBoost Randomized Search
best_xgb_model = random_search_xgb.best_estimator_

# Stacking with XGBoost, RandomForest, CatBoost, GradientBoosting, LightGBM, and AdaBoost
base_learners = [
    ('xgb', best_xgb_model),
    ('rf', rf_model),
    ('catboost', catboost_model),
    ('gb', gb_model),
    ('lgb', lgb_model),
    ('ada', ada_model)
]

final_estimator = Ridge()

# Stacking Regressor
stacking_model = StackingRegressor(estimators=base_learners, final_estimator=final_estimator)
stacking_model.fit(X_train_selected, y_train)

# Predictions and evaluation
y_pred_stack = stacking_model.predict(X_test_selected)
test_mse_stack = mean_squared_error(np.expm1(y_test), np.expm1(y_pred_stack))  # Convert back from log scale
r2_stack = r2_score(np.expm1(y_test), np.expm1(y_pred_stack))

print(f"Stacking Model Test MSE: {test_mse_stack}")
print(f"Stacking Model R2 Score: {r2_stack}")

# Voting Regressor with XGBoost, RandomForest, and CatBoost
voting_model = VotingRegressor(estimators=[('xgb', best_xgb_model), ('rf', rf_model), ('catboost', catboost_model)])
voting_model.fit(X_train_selected, y_train)

# Predictions for voting model
y_pred_voting = voting_model.predict(X_test_selected)
test_mse_voting = mean_squared_error(np.expm1(y_test), np.expm1(y_pred_voting))  # Convert back from log scale
r2_voting = r2_score(np.expm1(y_test), np.expm1(y_pred_voting))

print(f"Voting Model Test MSE: {test_mse_voting}")
print(f"Voting Model R2 Score: {r2_voting}")

# Cross-validation scores for stacking model
cv_scores = cross_val_score(stacking_model, X_train_selected, y_train, cv=5, scoring='r2')
print(f"Cross-validated R2 scores: {cv_scores}")

# Plot Actual vs Predicted (for Stacking Model)
plt.figure(figsize=(8, 6))
plt.scatter(np.expm1(y_test), np.expm1(y_pred_stack), alpha=0.5)
plt.plot([min(np.expm1(y_test)), max(np.expm1(y_test))], [min(np.expm1(y_test)), max(np.expm1(y_test))], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Stacking Model)")
plt.show()

# Residual analysis
residuals = np.expm1(y_test) - np.expm1(y_pred_stack)
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution (Stacking Model)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.axvline(0, color='red', linestyle='--')
plt.show()
