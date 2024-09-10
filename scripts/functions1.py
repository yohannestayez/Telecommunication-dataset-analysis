
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from IPython.display import display
from scipy import stats


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




def remove_duplicates_and_undefined(df):
    """
    Removes duplicates and rows with 'undefined' values in Handset Type and Handset Manufacturer.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df.drop_duplicates(inplace=True)
    df = df.dropna(subset=['Handset Type', 'Handset Manufacturer'])
    
    # Remove rows where Handset Type or Handset Manufacturer is 'undefined'
    df = df[~df['Handset Type'].str.lower().isin(['undefined'])]
    df = df[~df['Handset Manufacturer'].str.lower().isin(['undefined'])]
    
    # Reset index
    df.reset_index(drop=True, inplace=True)
    
    return df

def handle_missing_values(df):
    """
    Handles missing values by filling numeric columns with mean and categorical columns with mode.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    
    Returns:
    pd.DataFrame: DataFrame with missing values handled.
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    categorical_columns = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def handle_outliers(df):
    """
    Handles outliers using Z-score and IQR methods.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Z-score method
    z_scores = np.abs(stats.zscore(df[numeric_columns]))
    df = df[(z_scores < 3).all(axis=1)]  # Keeping data points with z-scores < 3

    # IQR method
    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[numeric_columns] < (Q1 - 1.5 * IQR)) | (df[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    return df

def check_invalid_data(df):
    """
    Checks for invalid data (negative values in specific columns) and removes them.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to check.
    
    Returns:
    pd.DataFrame: DataFrame with invalid data removed.
    """
    invalid_data = df[(df['Total DL (Bytes)'] < 0) | (df['Total UL (Bytes)'] < 0)]
    print(f"Found {len(invalid_data)} rows with invalid data (negative values)")

    # Remove invalid data
    df = df[(df['Total DL (Bytes)'] >= 0) & (df['Total UL (Bytes)'] >= 0)]
    
    return df

# Main function to clean data

def clean_data(df):
    """
    Cleans the input DataFrame by removing duplicates, handling missing values, outliers, 
    invalid data, and resetting the index.

    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df = remove_duplicates_and_undefined(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = check_invalid_data(df)

    print("Data cleaning completed. Cleaned data is ready for analysis.")
    return df




def top_handsets_used(df):
    """
    Get the top 10 handsets used by customers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame containing handset data.
    
    Returns:
    pd.DataFrame: DataFrame with top handsets and their usage counts.
    """
    top_handsets = df.groupby('Handset Type')['IMEI'].count().nlargest(10).reset_index()
    top_handsets.columns = ['handset_model', 'usage_count']
    print("Top 10 Handsets Used by Customers:")
    display(top_handsets)  # Display as a table
    return top_handsets

def top_handset_manufacturers(df):
    """
    Get the top 3 handset manufacturers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame containing handset data.
    
    Returns:
    pd.DataFrame: DataFrame with top manufacturers and their usage counts.
    """
    top_manufacturers = df.groupby('Handset Manufacturer')['IMEI'].count().nlargest(3).reset_index()
    top_manufacturers.columns = ['manufacturer', 'usage_count']
    print("Top 3 Handset Manufacturers:")
    display(top_manufacturers)  # Display as a table
    return top_manufacturers




def visualize_results(top_handsets, top_manufacturers, top_5_handsets_per_manufacturer):
    """
    Visualize the results of the analysis.
    
    Parameters:
    top_handsets (pd.DataFrame): DataFrame with top handsets used.
    top_manufacturers (pd.DataFrame): DataFrame with top manufacturers.
    top_5_handsets_per_manufacturer (pd.DataFrame): DataFrame with top handsets per manufacturer.
    """
    # a. Top 10 Handsets Used by Customers
    plt.figure(figsize=(10, 6))
    sns.barplot(x='usage_count', y='handset_model', data=top_handsets)
    plt.title('Top 10 Handsets Used by Customers')
    plt.xlabel('Usage Count')
    plt.ylabel('Handset Model')
    plt.show()

    # b. Top 3 Handset Manufacturers
    plt.figure(figsize=(8, 5))
    sns.barplot(x='usage_count', y='manufacturer', data=top_manufacturers)
    plt.title('Top 3 Handset Manufacturers')
    plt.xlabel('Usage Count')
    plt.ylabel('Manufacturer')
    plt.show()

    # c. Top 5 Handsets per Top 3 Manufacturers
    plt.figure(figsize=(12, 8))
    sns.barplot(x='usage_count', y='handset_model', hue='manufacturer', data=top_5_handsets_per_manufacturer)
    plt.title('Top 5 Handsets per Top 3 Manufacturers')
    plt.xlabel('Usage Count')
    plt.ylabel('Handset Model')
    plt.legend(title='Manufacturer')
    plt.show()


def user_behavior_on_applications(df):
    """
    Aggregate user behavior on applications.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame containing application data.
    
    Returns:
    pd.DataFrame: DataFrame with aggregated user behavior metrics.
    """
    app_behavior = df.groupby(['IMEI']).agg(
        total_sessions=('IMEI', 'count'),
        total_session_duration=('Dur. (ms)', 'sum'),
        total_download=('Total DL (Bytes)', 'sum'),
        total_upload=('Total UL (Bytes)', 'sum'),
    ).reset_index()
    
    print("User Behavior on Applications:")
    display(app_behavior.head())
    return app_behavior



def describe_variables(df):
    """Describe all relevant variables in the DataFrame."""
    info = df.info()  # Describes data types
    stats = df.describe()  # Provides basic stats for numerical variables
    dtypes = df.dtypes
    print("Data Types:")
    display(dtypes)
    print("Descriptive Statistics:")
    display(stats)

def transform_variables(df):
    """Segment users into decile classes and compute total data per class."""
    df['Total_Session_Duration'] = df['Dur. (ms)']
    df['Decile_Class'] = pd.qcut(df['Total_Session_Duration'], 5, labels=[1, 2, 3, 4, 5])
    df['Total_Data'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
    
    decile_data = df.groupby('Decile_Class')['Total_Data'].sum().reset_index()
    print("Total data per decile class:")
    display(decile_data)

def basic_metrics(df):
    """Compute and print basic metrics for specified columns."""
    metrics = df[['Total_Session_Duration', 'Total DL (Bytes)', 'Total UL (Bytes)']].describe()
    print("\nBasic Metrics:")
    display(metrics)

def dispersion_analysis(df):
    """Compute and print dispersion metrics for specified columns."""
    dispersion_metrics = pd.DataFrame({
        'Variance': df[['Total_Session_Duration', 'Total DL (Bytes)', 'Total UL (Bytes)']].var(),
        'Standard Deviation': df[['Total_Session_Duration', 'Total DL (Bytes)', 'Total UL (Bytes)']].std(),
        'Range (Max - Min)': df[['Total_Session_Duration', 'Total DL (Bytes)', 'Total UL (Bytes)']].apply(np.ptp)
    })
    print("\nDispersion Metrics:")
    display(dispersion_metrics)

def univariate_analysis(df):
    """Perform graphical univariate analysis."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['Total_Session_Duration'], kde=True, bins=20)
    plt.title('Histogram of Session Duration')

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df['Total_Session_Duration'])
    plt.title('Boxplot of Session Duration')
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.histplot(df['Total_Data'], kde=True, bins=20)
    plt.title('Histogram of Total Data (Download + Upload)')
    plt.show()

def bivariate_analysis(df, app_columns):
    """Create scatter plots for Total Data vs. app usage."""
    num_apps = len(app_columns)
    num_rows = (num_apps + 1) // 2  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 5 * num_rows))  # Adjust the figsize for better visibility

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    for ax, app in zip(axes, app_columns):
        sns.scatterplot(x=df[app], y=df['Total_Data'], ax=ax)
        ax.set_title(f'Total Data vs {app}')
        ax.set_xlabel(app)
        ax.set_ylabel('Total Data')

    # Hide any unused subplots
    for i in range(num_apps, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()  # Adjusts spacing to prevent overlap
    plt.show()

def correlation_analysis(df, app_columns):
    """Compute and plot the correlation matrix for app usage data."""
    app_data = df[app_columns]
    corr_matrix = app_data.corr()
    
    print("\nCorrelation Matrix:")
    display(corr_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix for App Usage Data')
    plt.show()

def pca_analysis(df, app_columns):
    """Perform PCA and plot the results."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[app_columns])

    pca = PCA(n_components=2)  # Reduce to 2 components for simplicity
    pca_result = pca.fit_transform(scaled_data)

    # Create a DataFrame with PCA results
    pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
    
    print("Explained Variance Ratio by PCA Components:")
    display(pd.DataFrame(pca.explained_variance_ratio_, columns=['Explained Variance']))

    # Scatter plot of PCA result
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_df['PCA1'], y=pca_df['PCA2'])
    plt.title('PCA of App Usage Data')
    plt.show()

