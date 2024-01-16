import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from src.utility.utils import load_data, return_next_item
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def if_satisfied(row):
    if row['energyconsumption'] < 12.9 and row['packetloss'] < 10 and row['latency'] < 5:
        return 1
    else:
        return 0
    

def getdata():
    lst_X = []
    lst_y = []
    lst_train, lst_test = load_data(
        r'D:\projects\RL-DeltaIoT\data\DeltaIoTv1\train')

    df_handler = return_next_item(lst_train, normalize=False)

    try:
        while (True):
            df = next(df_handler).drop('verification_times', axis=1)
            scaler = StandardScaler()
            df_X = pd.DataFrame(scaler.fit_transform(df['features'].values.tolist()))
            lst_X.append(df_X)
            # df['y'] = 1 if ((df['energyconsumption'] < 12.9) & (df['packetloss'] < 10) & (df['latency'] < 5)).all() else 0
            df['y'] = df.apply(lambda x: if_satisfied(x), axis=1)
            # df['y'] = 1 if ((df['energyconsumption'] < 12.9) & (df['packetloss'] < 10) & (df['latency'] < 5)) else 0
            lst_y.append(df['y'])

    except:
        X = pd.concat(lst_X)
        y = pd.concat(lst_y)

    return X, y


def kmeans_analysis(X, n_clusters=4):

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    # Get cluster centers and labels
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Plot the data points and cluster centers
    plt.scatter(X[:, 0], X[:, 1], c=labels,
                cmap='viridis', alpha=0.7, edgecolors='k')
    plt.scatter(centers[:, 0], centers[:, 1], c='red',
                marker='X', s=200, label='Cluster Centers')
    plt.title('K-means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
    print('f')


def pca_analysis(X, y):

    # Apply PCA to reduce the data to 2 principal components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    return X_pca, y

if __name__ == '__main__':
    X, y = getdata()
    X , y = pca_analysis(X, y)
    
    kmeans_analysis(X, n_clusters=4)


