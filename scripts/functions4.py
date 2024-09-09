# scripts/functions.py
import psycopg2
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.impute import SimpleImputer


def load_data():
    conn = psycopg2.connect(
        host="localhost",
        database="telecom",
        user="postgres",
        password="root"
    )
    query = "SELECT * FROM xdr_data"  # Adjust table name
    df = pd.read_sql(query, conn)
    conn.close()

    
    return df




# Step 1: Connect to the PostgreSQL database and load the telecom data


def clean_data(df):
    # Remove rows with missing target values
    df = df.dropna(subset=['Activity Duration DL (ms)', 'Activity Duration UL (ms)', 'Total DL (Bytes)', 
                           'Total UL (Bytes)', 'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'TCP DL Retrans. Vol (Bytes)', 
                           'TCP UL Retrans. Vol (Bytes)', 'Handset Type'])
    return df


# Step 2: Data Preprocessing - Scaling the data
def scale_data(df, engagement_columns, experience_columns):
    scaler = StandardScaler()

    # Scale engagement data
    scaled_engagement_data = scaler.fit_transform(df[engagement_columns])

    # Scale experience data
    scaled_experience_data = scaler.fit_transform(df[experience_columns])

    return scaled_engagement_data, scaled_experience_data

# Step 3: Perform KMeans clustering for engagement and experience analysis
def perform_kmeans_clustering(data, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data)
    return kmeans.labels_, kmeans.cluster_centers_

# Step 4: Calculate Euclidean distance for engagement and experience scores
def calculate_scores(df, engagement_data, experience_data, engagement_centroid, experience_centroid):
    df['Engagement_Score'] = pairwise_distances(engagement_data, [engagement_centroid], metric='euclidean')
    df['Experience_Score'] = pairwise_distances(experience_data, [experience_centroid], metric='euclidean')
    df['Satisfaction_Score'] = (df['Engagement_Score'] + df['Experience_Score']) / 2
    return df




# Step 5: Train a regression model to predict satisfaction score
def train_regression_model(df):
    # Define feature columns and target column
    feature_columns = df.columns[df.columns != 'Satisfaction_Score']
    X = df[feature_columns]
    y = df['Satisfaction_Score']
    
    # Separate numeric and non-numeric columns in X
    numeric_cols = X.select_dtypes(include=['number']).columns
    non_numeric_cols = X.select_dtypes(exclude=['number']).columns
    
    # Impute missing values in numeric features using 'mean' strategy
    X_numeric = X[numeric_cols]
    X_imputer = SimpleImputer(strategy='mean')
    X_numeric_imputed = X_imputer.fit_transform(X_numeric)
    
    # For simplicity, dropping non-numeric columns 
    X_clean = pd.DataFrame(X_numeric_imputed, columns=numeric_cols)
    
    # Impute missing values in the target variable 'Satisfaction_Score'
    y_imputer = SimpleImputer(strategy='mean')
    y_imputed = y_imputer.fit_transform(y.values.reshape(-1, 1)).ravel()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_imputed, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    return model

# Step 6: Perform KMeans on the engagement & experience scores
def kmeans_on_scores(df, n_clusters=2):
    scores = df[['Engagement_Score', 'Experience_Score']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df['Cluster'] = kmeans.fit_predict(scores)
    return df

def export_to_postgres(df):
    engine = create_engine('postgresql://postgres:root@localhost:5432/telecom')

    df[['Bearer Id', 'MSISDN/Number', 'Handset Type', 'Engagement_Score', 'Experience_Score', 'Satisfaction_Score']].to_sql(
        'satisfaction_scores', engine, if_exists='replace', index=False
    )

    # Step 8: Model Deployment and Tracking using MLflow
def deploy_model_with_tracking(model, X_train, y_train):
    mlflow.start_run()

    mlflow.log_param("model_type", "Linear Regression")
    mlflow.log_param("features", ['Engagement_Score', 'Experience_Score'])

    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "regression_model")

    mlflow.end_run()
