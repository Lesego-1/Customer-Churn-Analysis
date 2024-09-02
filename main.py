import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# Initiaize DataFrame
df = pd.read_csv("Customer_Churn.csv")

# Print data types for each column
column_dtypes = df.dtypes
print(column_dtypes)
print()

# Check for missing values
missing_values = df.isnull().sum().sum()
print(f"Total Missing Values {missing_values}") # There are no missing values

# Convert TotalCharges column to numerical format
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values with median
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Check for duplicate rows
duplicate_rows = df.duplicated().sum()
print(f"Duplicate Rows: {duplicate_rows}") # There are no duplicate rows

# Replaec all yes and no with 1 and 0
df = df.replace({"Yes":1, "No":0, "No internet service":0, "No phone service":0})

# Initialize numerical columns
numerical_columns = [df["tenure"], df["MonthlyCharges"], df["TotalCharges"]]

# Plot boxplot for each
for col in numerical_columns:
    plt.figure(figsize=(12,6))
    sns.boxplot(col)
    plt.title(f"Boxplot for {col.name}")
    
    # Create ylabel for each
    if col.name == df["tenure"].name:
        plt.ylabel("Amount of months")
    else:
        plt.ylabel("Amount ($)")
        
    plt.show() # There are no outliers
    
# Replace customerID values with indexes
df["customerID"] = df.index

# One-Hot encode data
df = pd.get_dummies(df)

# Rename column for clarity
df = df.rename(columns={"InternetService_0":"No_Internet_Service"})

# Initialize scaler
scaler = StandardScaler()
# Create new DataFrame with scaled data
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

scaled_df.head()

# Initialize correlation matrix
corr_matrix = scaled_df.corr()

# Heatmap of cleaned data
plt.figure(figsize=(12,6))
sns.heatmap(corr_matrix, cmap="viridis")
plt.title("Heatmap of cleaned data")
plt.show()

# Make target column non-numeric (0's and 1's)
scaled_df["Churn"] = scaled_df["Churn"].replace({-0.6010234796064696: 0, 1.6638285090871447: 1})
# Convert target values into int
scaled_df["Churn"] = scaled_df["Churn"].astype(int)

# Initialize features and target
X = scaled_df.drop(columns=["Churn"])
y = scaled_df["Churn"]

# Initialize train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=42)

# Create dummy model for feature selection
dummy_model = RandomForestClassifier(random_state=42)
dummy_model.fit(X_train, y_train) # Fit data into model

# Apply feature selection
selector = SelectFromModel(dummy_model)
selector.fit_transform(X_train, y_train) # Fit data into selector

# Selected features
X_test_selected = selector.transform(X_test)

# Create model
model = RandomForestClassifier(random_state=42)

# Parameter Grid
param_grid = {
    'n_estimators' : [100, 200],
    'criterion' : ["gini", "entropy"],
    'max_depth' : [None, 10],
    "min_samples_split" : [2, 5],
    "min_samples_leaf" : [1, 5],
    "max_features" : ["auto", "sqrt"],
    "bootstrap" : [True, False],
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, cv=2, param_grid=param_grid, verbose=2, n_jobs=-1)
# Fit data into GridSearchCV
grid_search.fit(X_train, y_train)

# Print best parameters for the model
best_params = grid_search.best_params_

print(f"Best parameters for the model: {best_params}")

# Initialize model with best parameters
best_rf_clf = grid_search.best_estimator_
best_rf_clf.fit(X_train, y_train)
# Predict the y values
y_test_pred = best_rf_clf.predict(X_test)

# Classification Report
clf_report = classification_report(y_test, y_test_pred)

print(clf_report)

# Get feature importances
feature_importances = pd.Series(best_rf_clf.feature_importances_, index=X_train.columns)
# Sort feature importances
sorted_importances = feature_importances.sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x=sorted_importances, y=sorted_importances.index)
plt.title("Feature Importances in Random Forests Classifier")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()