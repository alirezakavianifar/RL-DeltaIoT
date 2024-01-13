import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from src.utility.utils import load_data, return_next_item


def getdata():
    lst = []
    lst_train, lst_test = load_data(
        r'D:\projects\RL-DelataIoT-fortest\RL-DeltaIoT\data\DeltaIoTv1\train')

    df_handler = return_next_item(lst_train)

    try:
        while (True):
            df = next(df_handler)['features']
            lst.append(pd.DataFrame(df.values.tolist()))
    except:
        X = pd.concat(lst)

    return X


def k_means():

    X = getdata()
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X)

    # Get cluster centers and labels
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Plot the data points and cluster centers
    plt.scatter(X.iloc[:, 0], X.iloc[:, 19], c=labels,
                cmap='viridis', alpha=0.7, edgecolors='k')
    plt.scatter(centers[:, 0], centers[:, 1], c='red',
                marker='X', s=200, label='Cluster Centers')
    plt.title('K-means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
    print('f')


def pca_analysis():

    # Load the Iris dataset
    X = getdata()

    # Apply PCA to reduce the data to 2 principal components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Visualize the original and reduced-dimensional data
    plt.figure(figsize=(12, 5))

    # Plot the original data
    plt.subplot(1, 2, 1)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1])
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # Plot the reduced-dimensional data after PCA
    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.title('Data after PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # pca_analysis()
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.datasets import load_iris

    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Apply PCA to reduce the data to 2 principal components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Visualize the original and reduced-dimensional data
    plt.figure(figsize=(12, 5))

    # Plot the original data
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # Plot the reduced-dimensional data after PCA
    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolors='k')
    plt.title('Data after PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.tight_layout()
    plt.show()
    print('f')

