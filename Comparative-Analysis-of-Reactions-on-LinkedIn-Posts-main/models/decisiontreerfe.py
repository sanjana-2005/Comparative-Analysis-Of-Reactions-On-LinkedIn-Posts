# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
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
data['followers'] = data['followers'].astype(str).str.replace(',', '', regex=False)
data['followers'] = pd.to_numeric(data['followers'], errors='coerce')
data['followers'].fillna(data['followers'].median(), inplace=True)

# Impute categorical columns with mode
for col in ["headline", "location", "content", "media_type"]:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Drop columns with high percentages of missing values
data.drop(columns=["views", "votes"], inplace=True)

# Handle 'connections' column if exists
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

# Visualize outliers using box plots
for col in ['num_hashtags', 'reactions', 'comments', 'followers']:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot for {col}')
    plt.show()

# Feature Engineering
data['followers_connections_interaction'] = data['followers'] * data['connections']
data['comments_hashtags_interaction'] = data['comments'] * data['num_hashtags']

# Log-transform skewed numerical columns
data['log_followers'] = np.log1p(data['followers'])
data['log_reactions'] = np.log1p(data['reactions'])

# Define features (X) and target (y)
X = data[['headline', 'location', 'log_followers', 'connections', 'media_type', 
           'num_hashtags', 'comments', 'followers_connections_interaction', 
           'comments_hashtags_interaction']]
y = data['log_reactions']  # Log-transformed target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode categorical features
label_encoders = {}
for col in ['headline', 'location', 'media_type']:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))  # Ensure all are strings
    X_test[col] = le.transform(X_test[col].astype(str))  # Use the same encoder for test set
    label_encoders[col] = le

# Define the preprocessing steps
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numeric_features),
        ('cat', 'passthrough', categorical_features)  # Categorical can be label encoded
    ])

# Define the Decision Tree Regressor model for RFE
decision_tree_model = DecisionTreeRegressor(random_state=42)

# Create RFE wrapper around the decision tree model
num_features_to_select = 5  # You can adjust this
rfe = RFE(estimator=decision_tree_model, n_features_to_select=num_features_to_select)

# Create a pipeline that includes preprocessing, RFE, and the decision tree model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', rfe),
    ('model', decision_tree_model)
])

# Fit the model
pipeline.fit(X_train, y_train)

# Predictions and evaluation
y_pred = pipeline.predict(X_test)
test_mse = mean_squared_error(np.expm1(y_test), np.expm1(y_pred))  # Convert back from log scale
r2 = r2_score(np.expm1(y_test), np.expm1(y_pred))

print(f"Decision Tree Model with RFE Test MSE: {test_mse}")
print(f"Decision Tree Model with RFE R2 Score: {r2}")

# Cross-validation scores
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
print(f"Cross-validated R2 scores: {cv_scores}")

# Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(np.expm1(y_test), np.expm1(y_pred), color='blue', edgecolors='black', alpha=0.6)
plt.plot([min(np.expm1(y_test)), max(np.expm1(y_test))], 
         [min(np.expm1(y_test)), max(np.expm1(y_test))], 
         color='red', linestyle='--')
plt.title('Actual vs Predicted Reactions (Decision Tree Model with RFE)')
plt.xlabel('Actual Reactions')
plt.ylabel('Predicted Reactions')
plt.show()

# Residuals Plot
plt.figure(figsize=(8, 6))
sns.residplot(x=np.expm1(y_test), y=np.expm1(y_pred), color="g")
plt.title('Residuals Plot (Decision Tree Model with RFE)')
plt.xlabel('Actual Reactions')
plt.ylabel('Residuals')
plt.show()
