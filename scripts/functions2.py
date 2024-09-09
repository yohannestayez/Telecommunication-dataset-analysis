import psycopg2
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from IPython.display import display




def fetch_data_from_db(dbname, user, password, host, query):
    """
    Fetch data from a PostgreSQL database and return it as a pandas DataFrame.

    Parameters:
    dbname (str): Name of the PostgreSQL database.
    user (str): Username for the database.
    password (str): Password for the database.
    host (str): Host address of the database server.
    query (str): SQL query to execute.

    Returns:
    pd.DataFrame: DataFrame containing the query results.
    """
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host
        )

        # Load the data into a pandas DataFrame
        df = pd.read_sql(query, conn)

    except Exception as e:
        print(f"Error: {e}")
        df = None

    finally:
        # Close the database connection
        if conn:
            conn.close()

    return df


def clean_and_transform_data(df):
    """
    Cleans and transforms the input DataFrame by renaming columns, handling missing values,
    creating new columns, and removing duplicates.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing telecom data.

    Returns:
    pd.DataFrame: Cleaned and transformed DataFrame.
    """
    # Rename columns for consistency and readability
    df.rename(columns={
        'Dur. (ms)': 'session_duration',
        'Total DL (Bytes)': 'total_dl_bytes',
        'Total UL (Bytes)': 'total_ul_bytes'
    }, inplace=True)

    # Drop rows where essential columns are missing
    df.dropna(subset=['MSISDN/Number'], inplace=True)

    # Fill missing values
    df['session_duration'].fillna(df['session_duration'].mean(), inplace=True)
    df['total_dl_bytes'].fillna(0, inplace=True)
    df['total_ul_bytes'].fillna(0, inplace=True)

    # Create a new column for total traffic (downlink + uplink)
    df['total_traffic'] = df['total_dl_bytes'] + df['total_ul_bytes']

    # Remove any duplicate rows
    df.drop_duplicates(inplace=True)

    return df

def aggregate_user_engagement(df):
    """
    Aggregates user engagement metrics per customer ID.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing telecom data.

    Returns:
    pd.DataFrame: DataFrame with aggregated metrics for each customer.
    """
    user_engagement = df.groupby('MSISDN/Number').agg({
        'session_duration': 'sum',
        'total_traffic': 'sum'
    }).reset_index()
    return user_engagement

def perform_clustering(user_engagement, n_clusters=3):
    """
    Normalizes engagement metrics and performs K-Means clustering.

    Parameters:
    user_engagement (pd.DataFrame): DataFrame with aggregated metrics for each customer.
    n_clusters (int): Number of clusters for K-Means. Default is 3.

    Returns:
    pd.DataFrame: DataFrame with cluster assignments.
    """
    # Normalize the engagement metrics
    scaler = StandardScaler()
    scaled_engagement = scaler.fit_transform(user_engagement[['session_duration', 'total_traffic']])

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    user_engagement['cluster'] = kmeans.fit_predict(scaled_engagement)

    return user_engagement

def summarize_clusters(user_engagement):
    """
    Computes summary metrics for each cluster.

    Parameters:
    user_engagement (pd.DataFrame): DataFrame with aggregated metrics and cluster assignments.

    Returns:
    pd.DataFrame: Summary metrics for each cluster.
    """
    cluster_summary = user_engagement.groupby('cluster').agg({
        'session_duration': ['min', 'max', 'mean', 'sum'],
        'total_traffic': ['min', 'max', 'mean', 'sum']
    })
    return cluster_summary

def plot_top_applications(df):
    """
    Plots the top 3 most used applications based on download data.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing application usage data.

    Returns:
    None
    """
    app_usage = df[['Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)']].sum()
    top_apps = app_usage.nlargest(3)

    plt.figure(figsize=(10, 6))
    top_apps.plot(kind='bar')
    plt.title('Top 3 Most Used Applications')
    plt.ylabel('Total Download Data (Bytes)')
    plt.xlabel('Application')
    plt.show()

def elbow_method(scaled_engagement, max_k=10):
    """
    Finds the optimized value of k using the Elbow method.

    Parameters:
    scaled_engagement (np.array): Scaled engagement metrics for clustering.
    max_k (int): Maximum number of clusters to check. Default is 10.

    Returns:
    None
    """
    sse = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(scaled_engagement)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), sse, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.show()
