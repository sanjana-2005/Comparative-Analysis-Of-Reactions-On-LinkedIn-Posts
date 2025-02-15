# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load dataset with correct encoding
file_path = 'C:/Users/slive/Downloads/MACHINE LEARNING/PROJECT/archive/company_data.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Handling missing values and data type conversion for 'followers'
if data['followers'].dtype != 'object':
    data['followers'] = data['followers'].astype(str)

data['followers'] = data['followers'].str.replace(',', '', regex=False)
data['followers'] = pd.to_numeric(data['followers'], errors='coerce')

# Impute numerical columns with median
data['followers'].fillna(data['followers'].median(), inplace=True)

# Impute categorical columns with mode
for col in ["headline", "location", "content", "media_type"]:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Drop columns with high percentages of missing values
data.drop(columns=["views", "votes"], inplace=True)

# Handle 'connections' column if it exists
if 'connections' in data.columns:
    data['connections'] = data['connections'].str.replace(',', '').astype(float)

# Outlier Detection and Removal using IQR
def whisker(col):
    q1, q3 = np.percentile(col, [25, 75])
    iqr = q3 - q1
    lw = q1 - (1.5 * iqr)
    uw = q3 + (1.5 * iqr)
    return lw, uw

# Apply whisker function for outlier detection
for col in ['num_hashtags', 'reactions', 'comments', 'followers']:
    lw, uw = whisker(data[col])
    data[col] = np.clip(data[col], lw, uw)

# Feature Engineering
data['followers_connections_interaction'] = data['followers'] * data['connections']
data['comments_hashtags_interaction'] = data['comments'] * data['num_hashtags']
data['followers_sq'] = data['followers'] ** 2
data['log_comments'] = np.log1p(data['comments'])
data['interaction_advanced'] = data['followers'] * data['num_hashtags'] * data['comments']

# Log-transform skewed numerical columns
data['log_followers'] = np.log1p(data['followers'])
data['log_reactions'] = np.log1p(data['reactions'])

# Define features (X) and target (y)
X = data[['headline', 'location', 'log_followers', 'connections', 'media_type', 
          'num_hashtags', 'comments', 'followers_connections_interaction', 
          'comments_hashtags_interaction', 'followers_sq', 'log_comments', 'interaction_advanced']]
y = data['log_reactions']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode categorical features
label_encoders = {}
for col in ['headline', 'location', 'media_type']:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Define preprocessing steps
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Column transformer for scaling numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numeric_features),
        ('cat', 'passthrough', categorical_features)
    ])

# Feature Selection: Use LinearSVR for RFE feature selection
lin_svr = LinearSVR(max_iter=10000)  # Estimator with `coef_` attribute

# Recursive Feature Elimination with LinearSVR
rfe = RFE(estimator=lin_svr, n_features_to_select=8)

# Define the pipeline with preprocessing, RFE (based on LinearSVR), and SVR
pipeline_svr_rfe = Pipeline(steps=[('preprocessor', preprocessor), 
                                   ('rfe', rfe), 
                                   ('svr', SVR(kernel='rbf'))])

# Train the pipeline
pipeline_svr_rfe.fit(X_train, y_train)

# Make predictions
y_pred_svr = pipeline_svr_rfe.predict(X_test)

# Convert back from log scale for evaluation
test_mse_svr = mean_squared_error(np.expm1(y_test), np.expm1(y_pred_svr))
r2_svr = r2_score(np.expm1(y_test), np.expm1(y_pred_svr))

# Print performance metrics for SVR with RFE
print(f"Support Vector Regression Model with RFE Test MSE: {test_mse_svr}")
print(f"Support Vector Regression Model with RFE R² Score: {r2_svr}")

# Plot Actual vs Predicted for SVR with RFE
plt.figure(figsize=(8, 6))
plt.scatter(np.expm1(y_test), np.expm1(y_pred_svr), color='blue', edgecolors='black', alpha=0.6)
plt.plot([min(np.expm1(y_test)), max(np.expm1(y_test))], 
         [min(np.expm1(y_test)), max(np.expm1(y_test))], 
         color='red', linestyle='--')
plt.title(f'Actual vs Predicted Reactions (SVR with RFE)')
plt.xlabel('Actual Reactions')
plt.ylabel('Predicted Reactions')
plt.show()

# Residuals Plot for SVR with RFE
plt.figure(figsize=(8, 6))
sns.residplot(x=np.expm1(y_test), y=np.expm1(y_pred_svr), lowess=True, color='blue')
plt.title(f'Residuals Plot (SVR with RFE)')
plt.xlabel('Actual Reactions')
plt.ylabel('Residuals')
plt.show()
