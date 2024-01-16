import pandas as pd
import glob
import tensorflow as tf
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm
from collections import defaultdict
import shutil
from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def measure_time(*args, **kwargs):

        before = time.process_time()
        result = func(*args, **kwargs)
        after = time.process_time()
        total_time = after-before
        print("The operation for %s took %s" %
              (func.__name__, total_time))
        return result, total_time
    return measure_time

def pca_analysis(X, y=None):
    """
    Perform Principal Component Analysis (PCA) on the input data.

    Parameters:
    - X: array-like or pd.DataFrame
        The input data for PCA.
    - y: array-like or None, optional (default=None)
        Target variable. If provided, it is returned as is in the output.
        If not provided, it remains None.

    Returns:
    - X_pca: array-like
        The reduced data matrix with 2 principal components.
    - y: array-like or None
        The target variable if provided. Otherwise, None.

    Example:
    X_train_pca, y_train = pca_analysis(X_train, y_train)
    X_test_pca, y_test = pca_analysis(X_test, y_test)
    """

    # Apply PCA to reduce the data to 2 principal components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    return X_pca, y



def kmeans_analysis(X, n_clusters=4, plotting=False):
    """
    Perform K-means clustering on the input data.

    Parameters:
    - X: ndarray
        Input data matrix.
    - n_clusters: int, optional (default=4)
        Number of clusters to form.
    - plotting: bool, optional (default=False)
        Whether to plot the data points and cluster centers.

    Returns:
    - cluster_indices: dict
        A dictionary where keys are cluster labels and values are lists
        containing the indices of data points assigned to each cluster.
    """

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    # Get cluster centers and labels
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Initialize a dictionary to store indices for each cluster
    cluster_indices = {i: [] for i in range(n_clusters)}

    # Populate the dictionary with indices
    for i, label in enumerate(labels):
        cluster_indices[label].append(i)

    # Plot the data points and cluster centers if specified
    if plotting:
        plt.scatter(X[:, 0], X[:, 1], c=labels,
                    cmap='viridis', alpha=0.7, edgecolors='k')
        plt.scatter(centers[:, 0], centers[:, 1], c='red',
                    marker='X', s=200, label='Cluster Centers')
        plt.title('K-means Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()

    return cluster_indices

def set_log_dir(name, path):
    # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(path, name)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    return train_summary_writer

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def return_next_item(lst, normalize=True, normalize_cols=['energyconsumption', 'packetloss', 'latency']):
    '''
    A generator function which returns the next data frame from the given repository.

    Parameters:
    - lst (list): A list containing items, where each item represents data in JSON format.
    - normalize (bool, optional): A flag indicating whether to normalize certain columns in the DataFrame. Default is True.
    - normalize_cols (list, optional): A list of column names to be normalized if 'normalize' is True. Default includes 'energyconsumption', 'packetloss', and 'latency'.

    Returns:
    - DataFrame: The next data frame from the given repository, with optional normalization applied to specified columns.

    Usage:
    - Call this function in a loop to iterate through a list of items (JSON data) and receive a DataFrame for each iteration.
    
    Example:
    ```python
    data_repository = [...]  # List containing JSON data items
    data_generator = return_next_item(data_repository, normalize=True)
    
    for dataframe in data_generator:
        # Process the DataFrame as needed
        print(dataframe)
    ```

    Notes:
    - The function uses a generator approach, allowing the iteration over the list of JSON data items to yield a DataFrame at each step.
    - If normalization is enabled, the specified columns are normalized using Min-Max scaling.
    - Normalization is applied in-place to the DataFrame.
    '''
    for index, item in enumerate(lst):
        df = pd.read_json(item)
        if normalize:
            scaler = MinMaxScaler()
            for item in normalize_cols:
                df[item] = scaler.fit_transform(df[item].values.reshape(-1, 1))
        yield df

def utility(energy_coef, packet_coef, latency_coef, energy_consumption, packet_loss, latency):

    return (energy_coef * energy_consumption + packet_coef * packet_loss + latency_coef * latency)


def move_files(files, dst, *args, **kwargs):
    """
    Copy files from source to destination.

    Parameters:
    - files (list): List of source file paths to be copied.
    - dst (str): Destination directory where files will be copied.
    - *args: Variable positional arguments (not used in the function).
    - **kwargs: Variable keyword arguments (not used in the function).

    Returns:
    None

    Note:
    The function uses the `shutil.copyfile` method to copy each file from the source
    directory to the destination directory. The progress of the file copying process
    is displayed using the `tqdm` progress bar.

    Example:
    move_files(['/path/to/source/file1.txt', '/path/to/source/file2.txt'],
               '/path/to/destination/')
    """
    for file in tqdm(files, desc="Copying Files"):
        # Extract the file name from the full path
        file_name = file.rsplit(os.sep, 1)[1]

        # Create the final destination path
        final_dst = os.path.join(dst, file_name)

        # Copy the file to the destination
        shutil.copyfile(file, final_dst)

# Example usage:
# move_files(['/path/to/source/file1.txt', '/path/to/source/file2.txt'],
#            '/path/to/destination/')



def scale_data(data):
    '''
    This function scales the data
    '''
    data = data.to_numpy().reshape(-1, 1)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data)
    return data


# Create dummy data for DLASeR

def create_dummy(data):
    mu, sigma = 0, 0.01
    s = np.random.normal(mu, sigma)
    return data + s


def return_next_item(lst, normalize=True, normalize_cols=['energyconsumption', 'packetloss', 'latency']):
    '''
    A generator function which returns the next data frame from given repository
    '''
    for index, item in enumerate(lst):
        df = pd.read_json(item)
        if normalize:
            scaler = MinMaxScaler()
            for item in normalize_cols:
                df[item] = scale_data(df[item])
        yield df


def load_data(path=None, load_all=False, version='', shuffle=False, fraction=1.0, test_size=0.2, return_train_test=True):
    json_files = glob.glob(os.path.join(path, version, "*.json"))
    fraction_to_retrieve = int(len(json_files) * fraction)
    json_files = json_files[:fraction_to_retrieve]
    json_lst = []
    json__train_lst = []
    json__test_lst = []
    train_lst = []
    test_lst = []

    if shuffle:
        random.shuffle(json_files)

    if load_all:
        for f in json_files:
            df = pd.read_json(f)
            json_lst.append(df)
            # Merge all dataframes into one
            json_lst = pd.concat(json_lst)

            return json_lst

    if return_train_test:
        train_size = int((1-test_size) * len(json_files))
        train_lst = json_files[:train_size]
        test_lst = json_files[train_size:]
    else:
        train_lst = json_files
        # for f in train_lst:
        #     df = pd.read_json(f)
        #     json__train_lst.append(df)

        # for f in test_lst:
        #     df = pd.read_json(f)
        #     json__test_lst.append(df)

    return train_lst, test_lst


def get_chosen_model(model_dics, params, model_type):
    model_dics_ = defaultdict(list)
    for key, item in model_dics.items():
        # v = key.split('\\')[-1].split('_')[1]
        model_name = key.split('\\')[-1].split('-')[0]
        if model_type == "1":
            if key == os.path.join(os.getcwd(), 'models', f"{model_name}-n_games=*-lr={params['lr']}-eps_min={params['eps_dec']}-batch_size={params['batch_size']}-gamma={params['gamma']}-q_next"):
                model_dics_[key] = [[model_dics[key][0][2]]]
        elif model_type == "2":
            if key == os.path.join(os.getcwd(), 'models', f"{model_name}-lr={params['lr']}-eps_min={params['eps_min']}-batch_size={params['batch_size']}-gamma={params['gamma']}_*_steps.zip"):
                model_dics_[key] = item
    return model_dics_


def get_tts_qs(df, packet_thresh, latency_thresh, energy_thresh):
    try:
        if not df.loc[df['packetloss'] < packet_thresh].empty:
            df = df.loc[df['packetloss'] < packet_thresh]

        if not df.loc[df['latency'] < latency_thresh].empty:
            df = df.loc[df['latency'] < latency_thresh]

        if not df.loc[df['energyconsumption'] > (energy_thresh - 0.1)].empty:
            df = df.loc[df['energyconsumption']
                        > (energy_thresh - 0.1)]

        if not df.loc[df['energyconsumption'] < energy_thresh].empty:
            df = df.loc[df['energyconsumption'] < energy_thresh]

        if not df.loc[df['energyconsumption'] == df['energyconsumption'].min()].iloc[-1:, :].empty:
            df_tts_final = df.loc[df['energyconsumption']
                                  == df['energyconsumption'].min()].iloc[-1:, :]
    except:
        pass
    return df_tts_final


def get_tt_qs(df, packet_thresh, latency_thresh, energy_thresh):
    if not df.loc[df['packetloss'] < packet_thresh].empty:
        df = df.loc[df['packetloss'] < packet_thresh]
    else:
        df = df.loc[df['packetloss'] == df['packetloss'].min()]

    if not df.loc[df['latency'] < latency_thresh].empty:
        df = df.loc[df['latency'] < latency_thresh]
    else:
        df = df.loc[df['latency'] == df['latency'].min()]

    df = df.loc[df['energyconsumption'] ==
                df['energyconsumption'].min()].head(1)

    return df


def scale_data(data):
    data = data.to_numpy().reshape(-1, 1)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data)
    return data
