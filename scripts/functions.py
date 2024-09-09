# scripts/functions.py

import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Database connection and loading data
def connect_to_db(dbname, user, password, host):
    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
    return conn

def load_data(query, conn):
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# 2. Data cleaning
def clean_data(df):
    df.drop_duplicates(inplace=True)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    categorical_columns = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    z_scores = np.abs(stats.zscore(df[numeric_columns]))
    df = df[(z_scores < 3).all(axis=1)]
    
    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[numeric_columns] < (Q1 - 1.5 * IQR)) | (df[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    df = df[(df['Total DL (Bytes)'] >= 0) & (df['Total UL (Bytes)'] >= 0)]
    return df



#3. Data analysis functions
def top_handsets(df):
    return df.groupby('Handset Type')['IMEI'].count().nlargest(10).reset_index()
def top_manufacturers(df):
    return df.groupby('Handset Manufacturer')['IMEI'].count().nlargest(3).reset_index()

def top_5_handsets_per_manufacturer(df, top_3_manufacturers):
     filtered_df = df[df['Handset Manufacturer'].isin(top_3_manufacturers)]
     return (filtered_df.groupby(['Handset Manufacturer', 'Handset Type'])['IMEI'].count()
             .groupby(level=0, group_keys=False)
             .nlargest(5).reset_index())
def app_behavior(df):
     return df.groupby(['IMEI']).agg(
         total_sessions=('IMEI', 'count'),
         total_session_duration=('Dur. (ms)', 'sum'),
         total_download=('Total DL (Bytes)', 'sum'),
         total_upload=('Total UL (Bytes)', 'sum'),
     ).reset_index()
# 4. Data visualization
 # Plotting top handsets
def plot_top_handsets(df):
    top_handsets_data = top_handsets(df)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='usage_count', y='handset_model', data=top_handsets_data)
    plt.title('Top 10 Handsets Used by Customers')
    plt.xlabel('Usage Count')
    plt.ylabel('Handset Model')
    plt.show()

 
def plot_top_manufacturers(df):
     top_manufacturers_data = top_manufacturers(df)
     plt.figure(figsize=(8, 5))
     sns.barplot(x='usage_count', y='manufacturer', data=top_manufacturers_data)
     plt.title('Top 3 Handset Manufacturers')
     plt.xlabel('Usage Count')
     plt.ylabel('Manufacturer')
     plt.show()
 # Plotting top 5 handsets per manufacturer










def get_top_10_handsets(df):
    top_handsets = df.groupby('Handset Type')['IMEI'].count().nlargest(10).reset_index()
    top_handsets.columns = ['handset_model', 'usage_count']
    return top_handsets

def get_top_3_manufacturers(df):
    top_manufacturers = df.groupby('Handset Manufacturer')['IMEI'].count().nlargest(3).reset_index()
    top_manufacturers.columns = ['manufacturer', 'usage_count']
    return top_manufacturers

def get_top_5_handsets_per_manufacturer(df, top_3_manufacturers):
    filtered_df = df[df['Handset Manufacturer'].isin(top_3_manufacturers)]
    top_5_handsets_per_manufacturer = (
        filtered_df.groupby(['Handset Manufacturer', 'Handset Type'])['IMEI'].count()
        .groupby(level=0, group_keys=False)
        .nlargest(5).reset_index()
    )
    top_5_handsets_per_manufacturer.columns = ['manufacturer', 'handset_model', 'usage_count']
    return top_5_handsets_per_manufacturer

def get_user_behavior(df):
    app_behavior = df.groupby(['IMEI']).agg(
        total_sessions=('IMEI', 'count'),
        total_session_duration=('Dur. (ms)', 'sum'),
        total_download=('Total DL (Bytes)', 'sum'),
        total_upload=('Total UL (Bytes)', 'sum'),
    ).reset_index()
    return app_behavior

def plot_top_handsets(top_handsets):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='usage_count', y='handset_model', data=top_handsets)
    plt.title('Top 10 Handsets Used by Customers')
    plt.xlabel('Usage Count')
    plt.ylabel('Handset Model')
    plt.show()

def plot_top_manufacturers(top_manufacturers):
    plt.figure(figsize=(8, 5))
    sns.barplot(x='usage_count', y='manufacturer', data=top_manufacturers)
    plt.title('Top 3 Handset Manufacturers')
    plt.xlabel('Usage Count')
    plt.ylabel('Manufacturer')
    plt.show()

def plot_top_5_handsets_per_manufacturer(top_5_handsets_per_manufacturer):
    plt.figure(figsize=(12, 8))
    sns.barplot(x='usage_count', y='handset_model', hue='manufacturer', data=top_5_handsets_per_manufacturer)
    plt.title('Top 5 Handsets per Top 3 Manufacturers')
    plt.xlabel('Usage Count')
    plt.ylabel('Handset Model')
    plt.legend(title='Manufacturer')
    plt.show()










# 5. PCA
def perform_pca(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
    return pca_df, pca.explained_variance_ratio_

def plot_pca(pca_df):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=pca_df['PCA1'], y=pca_df['PCA2'])
    plt.title('PCA of App Usage Data')
    plt.show()
