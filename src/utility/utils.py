import pandas as pd
import glob
import tensorflow as tf
import random
import os
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
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

def set_log_dir(name, path):
    # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(path, name)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    return train_summary_writer

def utility(energy_coef, packet_coef, latency_coef, energy_consumption, packet_loss, latency):

    return (energy_coef * energy_consumption + packet_coef * packet_loss + latency_coef * latency)

def move_files(files, dst, *args, **kwargs):
    """Copy files from src to destination"""
    for file in tqdm(files):
        file_name = file.rsplit('\\', 1)[1]
        final_dst = os.path.join(dst, file_name)
        shutil.copyfile(file, final_dst)

def scale_data(data):
    '''
    This function scales the data
    '''
    data = data.to_numpy().reshape(-1, 1)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data)
    return data


def return_next_item(lst, normalize=True, normalize_cols=['energyconsumption', 'packetloss', 'latency']):
    '''
    A generator function which returns the next data frame from given repository
    '''
    for item in lst:
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


def get_tts_qs(df, packet_thresh, latency_thresh, energy_thresh):
    if not df.loc[df['packetloss'] < packet_thresh].empty:
        df_tts = df.loc[df['packetloss'] < packet_thresh]

    if not df_tts.loc[df_tts['latency'] < latency_thresh].empty:
        df_tts = df_tts.loc[df_tts['latency'] < latency_thresh]

    if not df_tts.loc[df_tts['energyconsumption'] > (energy_thresh - 0.1)].empty:
        df_tts = df_tts.loc[df_tts['energyconsumption']
                            > (energy_thresh - 0.1)]

    if not df_tts.loc[df_tts['energyconsumption'] < energy_thresh].empty:
        df_tts = df_tts.loc[df_tts['energyconsumption'] < energy_thresh]

    if not df_tts.loc[df_tts['energyconsumption'] == df_tts['energyconsumption'].min()].iloc[-1:, :].empty:
        df_tts_final = df_tts.loc[df_tts['energyconsumption']
                                  == df_tts['energyconsumption'].min()].iloc[-1:, :]
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
