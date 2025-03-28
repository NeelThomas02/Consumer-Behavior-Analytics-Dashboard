import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the synthetic data
df = pd.read_csv('consumer_behavior_tech_revolutions.csv')

# Create the target variable based on the tech adoption score
df['tech_adoption_category'] = pd.cut(
    df['tech_adoption_score'],
    bins=[0, 0.4, 0.7, 1],
    labels=['low', 'medium', 'high'],
    include_lowest=True
)

# Remove rows with missing target values
df_clean = df.dropna(subset=['tech_adoption_category']).copy()

# Select only the four features for training
features = ['digital_literacy_score', 'tech_anxiety_score', 'annual_tech_purchases', 'avg_spend_per_item']
X = df_clean[features]
y = df_clean['tech_adoption_category']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Save the trained model and the scaler in the 'model/' folder
joblib.dump(knn, 'model/knn_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("Model training complete. Files saved in the 'model/' folder.")
