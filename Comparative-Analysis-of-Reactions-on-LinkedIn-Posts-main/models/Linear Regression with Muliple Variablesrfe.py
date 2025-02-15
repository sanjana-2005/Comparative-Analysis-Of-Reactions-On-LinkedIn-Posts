# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
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
# Check if the 'followers' column is a string and then replace commas
if data['followers'].dtype != 'object':  # Check if it's not already a string
    data['followers'] = data['followers'].astype(str)

data['followers'] = data['followers'].str.replace(',', '', regex=False)

# Convert to numeric after removing commas
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
    data[col] = np.clip(data[col], lw, uw)  # Replace outliers with upper/lower bounds

# Feature Engineering: Create new interaction terms and transformations
data['followers_connections_interaction'] = data['followers'] * data['connections']
data['comments_hashtags_interaction'] = data['comments'] * data['num_hashtags']
data['followers_sq'] = data['followers'] ** 2
data['log_comments'] = np.log1p(data['comments'])
data['interaction_advanced'] = data['followers'] * data['num_hashtags'] * data['comments']

# Log-transform skewed numerical columns
data['log_followers'] = np.log1p(data['followers'])
data['log_reactions'] = np.log1p(data['reactions'])

# Define features (X) and target (y) with new features
X = data[['headline', 'location', 'log_followers', 'connections', 'media_type', 
           'num_hashtags', 'comments', 'followers_connections_interaction', 
           'comments_hashtags_interaction', 'followers_sq', 'log_comments', 'interaction_advanced']]
y = data['log_reactions']  # Log-transformed target

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

# Initialize the linear regression model
linear_reg = LinearRegression()

# Initialize RFE to select top 8 features
rfe = RFE(estimator=linear_reg, n_features_to_select=8)

# Define the pipeline with preprocessing, PCA, RFE, and linear regression
pipeline_linear = Pipeline(steps=[
    ('preprocessor', preprocessor), 
    ('pca', PCA(n_components=8)), 
    ('feature_selection', rfe), 
    ('linear_regression', linear_reg)
])

# Train the model with RFE
pipeline_linear.fit(X_train, y_train)

# Make predictions
y_pred_linear = pipeline_linear.predict(X_test)

# Convert back from log scale for evaluation
test_mse_linear = mean_squared_error(np.expm1(y_test), np.expm1(y_pred_linear))
r2_linear = r2_score(np.expm1(y_test), np.expm1(y_pred_linear))

# Print performance metrics for Multiple Linear Regression with RFE
print(f"Multiple Linear Regression with RFE Model Test MSE: {test_mse_linear}")
print(f"Multiple Linear Regression with RFE Model R² Score: {r2_linear}")

# Plot Actual vs Predicted for Multiple Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(np.expm1(y_test), np.expm1(y_pred_linear), color='blue', edgecolors='black', alpha=0.6)
plt.plot([min(np.expm1(y_test)), max(np.expm1(y_test))], 
         [min(np.expm1(y_test)), max(np.expm1(y_test))], 
         color='red', linestyle='--')
plt.title(f'Actual vs Predicted Reactions (Multiple Linear Regression with RFE)')
plt.xlabel('Actual Reactions')
plt.ylabel('Predicted Reactions')
plt.show()

# Residuals Plot for Multiple Linear Regression
plt.figure(figsize=(8, 6))
sns.residplot(x=np.expm1(y_test), y=np.expm1(y_pred_linear), lowess=True, color='blue')
plt.title(f'Residuals Plot (Multiple Linear Regression with RFE)')
plt.xlabel('Actual Reactions')
plt.ylabel('Residuals')
plt.show()
