
import psycopg2
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from IPython.display import display


# Connect to PostgreSQL and import data from telecom.sql
def fetch_data(dbname, user, password, host, table_name):
    """
    Fetch all xDR records from the specified PostgreSQL database and return a DataFrame.

    Parameters:
    dbname (str): Database name.
    user (str): Username for the database.
    password (str): Password for the database user.
    host (str): Host where the database is located.
    table_name (str): Name of the table to fetch data from.

    Returns:
    pd.DataFrame: DataFrame containing the xDR records.
    """
    # Connect to the PostgreSQL database
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host
    )

    # Query to fetch all xDR records
    query = f"SELECT * FROM {table_name};"  # Adjust the table name as needed

    # Load the data into a pandas DataFrame
    df = pd.read_sql(query, conn)

    # Close the database connection
    conn.close()
    
    return df



def process_telecom_data(df):
    """
    Fills missing values and aggregates data per customer.

    Parameters:
    df (pd.DataFrame): DataFrame containing telecom data.

    Returns:
    pd.DataFrame: Aggregated DataFrame with averages per customer.
    """
    # Fill missing values with the mean for relevant columns
    df['TCP DL Retrans. Vol (Bytes)'].fillna(df['TCP DL Retrans. Vol (Bytes)'].mean(), inplace=True)
    df['TCP UL Retrans. Vol (Bytes)'].fillna(df['TCP UL Retrans. Vol (Bytes)'].mean(), inplace=True)
    df['Avg RTT DL (ms)'].fillna(df['Avg RTT DL (ms)'].mean(), inplace=True)
    df['Avg RTT UL (ms)'].fillna(df['Avg RTT UL (ms)'].mean(), inplace=True)
    df['Avg Bearer TP DL (kbps)'].fillna(df['Avg Bearer TP DL (kbps)'].mean(), inplace=True)
    df['Avg Bearer TP UL (kbps)'].fillna(df['Avg Bearer TP UL (kbps)'].mean(), inplace=True)

    # Aggregate per customer
    agg_data = df.groupby('MSISDN/Number').agg({
        'TCP DL Retrans. Vol (Bytes)': 'mean',
        'TCP UL Retrans. Vol (Bytes)': 'mean',
        'Avg RTT DL (ms)': 'mean',
        'Avg RTT UL (ms)': 'mean',
        'Avg Bearer TP DL (kbps)': 'mean',
        'Avg Bearer TP UL (kbps)': 'mean',
        'Handset Type': 'first'
    }).reset_index()

    # Rename columns for clarity
    agg_data.columns = ['MSISDN/Number', 'Avg TCP DL Retransmission', 'Avg TCP UL Retransmission', 
                        'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Throughput DL (kbps)', 
                        'Avg Throughput UL (kbps)', 'Handset Type']

    return agg_data

def analyze_tcp_retransmissions(agg_data):
    """
    Analyzes the top, bottom, and most frequent values for TCP DL and UL retransmissions.

    Parameters:
    agg_data (pd.DataFrame): Aggregated DataFrame with averages per customer.

    Returns:
    dict: A dictionary containing top 10, bottom 10, and most frequent TCP DL/UL retransmissions.
    """
    # Top 10 TCP DL and UL retransmissions
    top_tcp_dl = agg_data.nlargest(10, 'Avg TCP DL Retransmission')
    bottom_tcp_dl = agg_data.nsmallest(10, 'Avg TCP DL Retransmission')
    most_frequent_tcp_dl = agg_data['Avg TCP DL Retransmission'].mode()

    top_tcp_ul = agg_data.nlargest(10, 'Avg TCP UL Retransmission')
    bottom_tcp_ul = agg_data.nsmallest(10, 'Avg TCP UL Retransmission')
    most_frequent_tcp_ul = agg_data['Avg TCP UL Retransmission'].mode()

    # Store results in a dictionary
    results = {
        "top_tcp_dl": top_tcp_dl,
        "bottom_tcp_dl": bottom_tcp_dl,
        "most_frequent_tcp_dl": most_frequent_tcp_dl,
        "top_tcp_ul": top_tcp_ul,
        "bottom_tcp_ul": bottom_tcp_ul,
        "most_frequent_tcp_ul": most_frequent_tcp_ul
    }

    return results


def analyze_throughput_per_handset(agg_data):
    """
    Analyzes the average downlink and uplink throughput per handset type.

    Parameters:
    agg_data (pd.DataFrame): Aggregated DataFrame with averages per customer.

    Returns:
    pd.DataFrame: DataFrame containing the average throughput per handset type.
    """
    # Group by 'Handset Type' and calculate mean throughput for DL and UL
    throughput_per_handset = agg_data.groupby('Handset Type').agg({
        'Avg Throughput DL (kbps)': 'mean',
        'Avg Throughput UL (kbps)': 'mean'
    }).reset_index()

    return throughput_per_handset


def prepare_clustering_data(agg_data):
    """
    Prepares the data for clustering by selecting relevant features.

    Parameters:
    agg_data (pd.DataFrame): Aggregated DataFrame with customer data.

    Returns:
    pd.DataFrame: DataFrame with selected features for clustering.
    """
    return agg_data[['Avg TCP DL Retransmission', 'Avg TCP UL Retransmission', 
                     'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 
                     'Avg Throughput DL (kbps)', 'Avg Throughput UL (kbps)']]

def normalize_data(clustering_data):
    """
    Normalizes the clustering data.

    Parameters:
    clustering_data (pd.DataFrame): DataFrame with features for clustering.

    Returns:
    np.ndarray: Scaled data array.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(clustering_data)

def apply_kmeans(scaled_data, n_clusters=3):
    """
    Applies K-means clustering to the scaled data.

    Parameters:
    scaled_data (np.ndarray): Normalized data for clustering.
    n_clusters (int): Number of clusters for K-means.

    Returns:
    pd.Series: Cluster labels for each data point.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    return kmeans.fit_predict(scaled_data)

def analyze_clusters(agg_data):
    """
    Analyzes and returns characteristics of each cluster.

    Parameters:
    agg_data (pd.DataFrame): DataFrame with cluster assignments and numeric data.

    Returns:
    pd.DataFrame: DataFrame containing mean values for each cluster.
    """
    numeric_columns = agg_data.select_dtypes(include='number').columns
    return agg_data.groupby('Cluster')[numeric_columns].mean()

def display_cluster_characteristics(agg_data, num_clusters):
    """
    Displays the characteristics of each cluster.

    Parameters:
    agg_data (pd.DataFrame): DataFrame containing cluster assignments.
    num_clusters (int): Number of clusters.
    """
    for cluster_num in range(num_clusters):
        cluster_info = agg_data[agg_data['Cluster'] == cluster_num].select_dtypes(include='number').mean()
        print(f"\nCluster {cluster_num} characteristics:")
        display(cluster_info)

def plot_elbow_method(scaled_data, max_clusters=10):
    """
    Plots the elbow method for determining the optimal number of clusters.

    Parameters:
    scaled_data (np.ndarray): Normalized data for clustering.
    max_clusters (int): Maximum number of clusters to check.
    """
    sse = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(scaled_data)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), sse, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.grid(True)
    plt.show()


