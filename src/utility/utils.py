import pandas as pd
import glob
import tensorflow as tf
import random
import os
import re
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
import traceback
import streamlit as st
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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

def display_error_message(e, context=""):
    error_message = f"""
    <div style="border:1px solid red; padding: 10px; border-radius: 5px; background-color: #ffe6e6;">
        <h4 style="color: red;">An error occurred</h4>
        <p><strong>Context:</strong> {context}</p>
        <p><strong>Error:</strong> {str(e)}</p>
        <details>
            <summary>Traceback</summary>
            <pre>{traceback.format_exc()}</pre>
        </details>
    </div>
    """
    st.markdown(error_message, unsafe_allow_html=True)

def get_env_parameters(env_name, algo_name):
    if env_name in ['DeltaIoTv1', 'DeltaIoTv2']:
        n_obs_space = 3
        n_actions = 216 if env_name == 'DeltaIoTv1' else 4096 
    else:
        n_obs_space = 1
        n_actions = 100 
        use_dict_obs_space = False

    use_dict_obs_space = True if algo_name == 'HER_DQN' else False

    return n_obs_space, n_actions, use_dict_obs_space

def load_and_prepare_data(data_dir, from_cycles=0, to_cycles = 1505, cofig_num=33):
    
    LST_PACKET = []
    LST_ENERGY = []
    LST_DATA, _ = load_data(path=data_dir, load_all=False, version='', shuffle=False, fraction=1.0, test_size=0.2, return_train_test=False)
    DATA = return_next_item(LST_DATA, normalize=False)
    LST_PACKET = []
    LST_ENERGY = []
    LST_LATENCY = []
    cycle_metrics = []
    try:
        while True:
            data_ = next(DATA)
            LST_PACKET.append(data_['packetloss'].iloc[cofig_num])
            LST_ENERGY.append(data_['energyconsumption'].iloc[cofig_num])
            LST_LATENCY.append(data_['latency'].iloc[cofig_num])
            cycle_metrics.append({
                'packetloss': data_['packetloss'],
                'energyconsumption': data_['energyconsumption'],
                'latency': data_['latency']
            })
    except:
        pass
    
    df = pd.DataFrame()
    for i, metrics in enumerate(cycle_metrics[from_cycles:to_cycles]):
        cycle_df = pd.DataFrame(metrics)
        cycle_df['cycle'] = i
        df = pd.concat([df, cycle_df])

    return LST_PACKET[from_cycles:to_cycles], LST_ENERGY[from_cycles:to_cycles], LST_LATENCY[from_cycles:to_cycles], df

def plot_latency_vs_packet_loss(st, LST_LATENCY, LST_PACKET, LST_ENERGY):
    configurations = list(range(1, len(LST_LATENCY) + 1))
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Latency', color=color)
    ax1.scatter(configurations, LST_LATENCY, color=color, label='Latency')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Packet Loss', color=color)
    ax2.plot(configurations, LST_PACKET, color=color, marker='o', linestyle='None', label='Packet Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Latency and Packet Loss for Different Configurations')
    ax1.grid(True)
    fig.tight_layout()
    st.pyplot(fig)


def plot_metrics_vs_configurations(st, LST_LATENCY, LST_PACKET, LST_ENERGY):
    configurations = list(range(1, len(LST_LATENCY) + 1))
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # Plot Latency
    color = 'tab:blue'
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Latency', color=color)
    ax1.scatter(configurations, LST_LATENCY, color=color, label='Latency')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title('Latency vs Configurations')
    ax1.grid(True)

    # Plot Packet Loss
    color = 'tab:red'
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Packet Loss', color=color)
    ax2.plot(configurations, LST_PACKET, color=color, marker='o', linestyle='None', label='Packet Loss')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_title('Packet Loss vs Configurations')
    ax2.grid(True)

    # Plot Energy
    color = 'tab:green'
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('Energy', color=color)
    ax3.plot(configurations, LST_ENERGY, color=color, marker='x', linestyle='None', label='Energy')
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.set_title('Energy vs Configurations')
    ax3.grid(True)

    fig.tight_layout()
    st.pyplot(fig)




def plot_adaptation_spaces(st, df, from_cycles=0, to_cycles=1505):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['packetloss'], df['energyconsumption'], s=1, alpha=0.5)
    plt.xlabel('Packet loss (%)')
    plt.ylabel('Energy Consumption (mC)')
    plt.title(f'Adaptation spaces from cycles {from_cycles} to {to_cycles} cycles')
    plt.xticks([0,10,20,30,40,50,60,70,80])
    st.pyplot(plt.gcf())


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



def scale_data(data, vals={'energy_thresh':12.9, 'packet_thresh':10, 'latency_thresh':5}):
    '''
    This function scales the data
    '''
    normalized_value = {}
    data = data.to_numpy().reshape(-1, 1)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data)
    for k, v in vals.item():
        normalized_value[k] = (v - 12) / (13 - 12)
    return data, normalized_value


# Create dummy data for DLASeR

def create_dummy(data):
    mu, sigma = 0, 0.01
    s = np.random.normal(mu, sigma)
    return data + s


class TrackedGenerator:
    def __init__(self, generator):
        self.generator, self.gen_copy = self._tee(generator)
        self.consumed_count = 0
        self.remaining_count = sum(1 for _ in self.gen_copy)

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self.generator)
        self.consumed_count += 1
        self.remaining_count -= 1
        return value

    def consumed_items(self):
        return self.consumed_count

    def remaining_items(self):
        return self.remaining_count

    def _tee(self, iterable):
        import itertools
        return itertools.tee(iterable)



def return_next_item(lst, normalize=True, normalize_cols=['energyconsumption', 'packetloss', 'latency'],
                      energy_thresh=12.9, packet_thresh=10, latency_thresh=5):
    '''
    A generator function which returns the next data frame from given repository
    '''
    for index, item in enumerate(lst):
        df = pd.read_json(item)
        if normalize:
            df = scale_data(df)
        yield df

# Iterate through each item in the DataFrame in a generator-like fashion
def iterate_dataframe(df):
    return df

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
            if key == os.path.join(os.getcwd(), 'models', f"{model_name}-policy={params['policy']}-lr={params['lr']}-eps_min={params['eps_min']}-batch_size={params['batch_size']}-gamma={params['gamma']}-exploration_fraction={params['exploration_fraction']}_*_steps.zip"):
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
    # data = data.to_numpy().reshape(-1, 1)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data.iloc[:,-3:])
    # data = pd.DataFrame(data)
    return data
def parse_log_directory_name(log_dir):
    pattern = re.compile(r'algo=(?P<algo>[^-]+)-goal=(?P<goal>[^-]+)-env=(?P<env>[^-]+)-policy=(?P<policy>[^-]+)-lr=(?P<lr>[^-]+)-batch_size=(?P<batch_size>[^-]+)-gamma=(?P<gamma>[^-]+)-total_timesteps=(?P<total_timesteps>[^-]+)-exploration_fraction=(?P<exploration_fraction>[^-]+)')
    match = pattern.search(log_dir)
    if match:
        return match.groupdict()
    return {}

def exponential_moving_average(data, alpha=0.01):
    ema = [data[0]]
    for value in data[1:]:
        ema.append(alpha * value + (1 - alpha) * ema[-1])
    return np.array(ema)

def smooth_scalar(values, weight=0.6):
    smoothed_values = []
    last = values[0]
    for point in values:
        smoothed_value = last * weight + (1 - weight) * point
        smoothed_values.append(smoothed_value)
        last = smoothed_value
    return np.array(smoothed_values)

def read_from_tensorboardlog(st, log_dirs, smoothed=True, smooth_factor=0.6, filtered=None, policies=None, 
                             tags=['rollout/ep_rew_mean', 'eval/mean_reward'], 
                             titles=['DQN', 'DQN']):
    for index, tag in enumerate(tags):
        fig, ax = plt.subplots(figsize=(14, 10))
        for log_dir in log_dirs:
            params = parse_log_directory_name(log_dir)
            label = ', '.join([f'{k}={v}' for k, v in params.items()])
            label = re.search(r'policy=([^,]+)', label).group(1)
            log_dir = os.path.join(log_dir, 'DQN_1')
            event_acc = EventAccumulator(log_dir)
            event_acc.Reload()

            scalar_data = {
                tag: [(event.step, event.value) for event in event_acc.Scalars(tag)]
                for tag in event_acc.Tags()['scalars']
            }

            if tag in scalar_data:
                data = scalar_data[tag]
                steps, values = zip(*data)
                if smoothed:
                    values = smooth_scalar(values, weight=smooth_factor)
                
                ax.plot(steps, values, label=label, linewidth=4)  # Make lines bolder

        ax.set_title(titles[index] if index < len(titles) else f'Tag: {tag}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Steps', fontsize=14, fontweight='bold')
        ax.set_ylabel('Mean Reward', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)  # Position legend outside the plot
        plt.tight_layout()
        st.pyplot(fig)
