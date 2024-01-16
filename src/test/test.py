import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from src.utility.utils import load_data, return_next_item
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.utility.utils import pca_analysis, kmeans_analysis


def if_satisfied(row):
    if row['energyconsumption'] < 12.9 and row['packetloss'] < 10 and row['latency'] < 5:
        return 1
    else:
        return 0


def getdata():
    lst_X = []
    lst_y = []
    lst_train, lst_test = load_data(
        r'D:\projects\RL-DelataIoT-fortest\RL-DeltaIoT\data\DeltaIoTv1\train')

    df_handler = return_next_item(lst_train, normalize=False)

    try:
        while (True):
            df = next(df_handler).drop('verification_times', axis=1)
            scaler = StandardScaler()
            df_X = pd.DataFrame(scaler.fit_transform(
                df['features'].values.tolist()))
            X, y = pca_analysis(df_X)
            kmeans_analysis(X, n_clusters=4)
            lst_X.append(df_X)
            # df['y'] = 1 if ((df['energyconsumption'] < 12.9) & (df['packetloss'] < 10) & (df['latency'] < 5)).all() else 0
            df['y'] = df.apply(lambda x: if_satisfied(x), axis=1)
            # df['y'] = 1 if ((df['energyconsumption'] < 12.9) & (df['packetloss'] < 10) & (df['latency'] < 5)) else 0
            lst_y.append(df['y'])

    except:
        X = pd.concat(lst_X)
        y = pd.concat(lst_y)

    return X, y


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

if __name__ == '__main__':
    X, y = getdata()
    X, y = pca_analysis(X, y)
    kmeans_analysis(X, n_clusters=4)
