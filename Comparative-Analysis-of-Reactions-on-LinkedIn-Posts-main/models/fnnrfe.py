# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# Load dataset with correct encoding
file_path = 'C:/Users/slive/Downloads/MACHINE LEARNING/PROJECT/archive/company_data.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Handling missing values and data type conversion for 'followers'
data['followers'] = data['followers'].astype(str)  # Convert to string
data['followers'] = data['followers'].str.replace(',', '', regex=False)  # Remove commas
data['followers'] = pd.to_numeric(data['followers'], errors='coerce')  # Convert to numeric
data['followers'].fillna(data['followers'].median(), inplace=True)  # Fill missing values

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
    data[col] = np.clip(data[col], lw, uw)

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

# Split the data into train and test sets with fixed random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Encode categorical features
label_encoders = {}
for col in ['headline', 'location', 'media_type']:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
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

# Preprocess the training data
X_train_processed = preprocessor.fit_transform(X_train)

# Preprocess the test data
X_test_processed = preprocessor.transform(X_test)

# Feature Selection using Recursive Feature Elimination (RFE) with SVR
svr = SVR(kernel='linear')  # SVR model with linear kernel
rfe = RFE(estimator=svr, n_features_to_select=5)  # Select top 5 features
X_train_rfe = rfe.fit_transform(X_train_processed, y_train)
X_test_rfe = rfe.transform(X_test_processed)  # Apply the same transformation to test set

# Build the neural network model
def create_model():
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train_rfe.shape[1],)))  # Adjust input size after RFE
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))  # Increased units and added L2
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))  # Increased Dropout rate
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))  # Increased units and added L2
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))  # Increased units and added L2
    model.add(layers.Dense(1))  # Output layer for regression
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error')  # Adjusted learning rate
    return model

# Create the model
model = create_model()

# Split training data into final train and validation sets
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(X_train_rfe, y_train, test_size=0.2, random_state=random_seed)

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit the model
history = model.fit(X_train_final, y_train_final, validation_data=(X_val_final, y_val_final), 
                    epochs=150, callbacks=[early_stopping])  # Adjusted epochs

# Predictions and evaluation
y_pred = model.predict(X_test_rfe)
test_mse = mean_squared_error(np.expm1(y_test), np.expm1(y_pred))  # Convert back from log scale
r2 = r2_score(np.expm1(y_test), np.expm1(y_pred))

print(f"Neural Network Model Test MSE: {test_mse:.4f}")
print(f"Neural Network Model R² Score: {r2:.4f}")

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(np.expm1(y_test), np.expm1(y_pred), color='blue', edgecolors='black', alpha=0.6)
plt.plot([min(np.expm1(y_test)), max(np.expm1(y_test))], 
         [min(np.expm1(y_test)), max(np.expm1(y_test))], 
         color='red', linestyle='--')
plt.title('Actual vs Predicted Reactions (Neural Network Model)')
plt.xlabel('Actual Reactions')
plt.ylabel('Predicted Reactions')
plt.show()

# Residuals Plot
plt.figure(figsize=(8, 6))
sns.residplot(x=np.expm1(y_test), y=np.expm1(y_pred), color="g")
plt.title('Residuals Plot (Neural Network Model)')
plt.xlabel('Actual Reactions')
plt.ylabel('Residuals')
plt.show()
