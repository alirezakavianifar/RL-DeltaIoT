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
from mpl_toolkits.mplot3d import Axes3D

from src.utility.utils import pca_analysis, kmeans_analysis


def plot_landscape(energy_consumption, packet_loss, latency):

# Generate synthetic data
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Create a 3D plot with a wireframe plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the wireframe landscape
    ax.plot_trisurf(energy_consumption, packet_loss, latency, cmap='viridis', edgecolor='black', linewidth=0.2)

    # Add labels and title
    ax.set_xlabel('Energy Consumption')
    ax.set_ylabel('Packet Loss')
    ax.set_zlabel('Latency')
    ax.set_title('Non-Smooth Latency Landscape in IoT Setting')

    # Show the plot
    plt.show()


    print('ff')


def if_satisfied(row):
    if row['energyconsumption'] < 12.9 and row['packetloss'] < 10 and row['latency'] < 5:
        return 1
    else:
        return 0


def getdata():
    lst_X = []
    lst_y = []
    lst_perf = []
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
        plot_landscape(df['energyconsumption'],
                           df['packetloss'], df['latency'])
        X = pd.concat(lst_X)
        y = pd.concat(lst_y)

    return X, y

if __name__ == '__main__':
    X, y = getdata()
#     X, y = pca_analysis(X, y)
#     kmeans_analysis(X, n_clusters=4)


