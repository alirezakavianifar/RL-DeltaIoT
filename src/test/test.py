from src.utility.utils import load_data, return_next_item
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sample data

DATA_DIR = r'D:\projects\gheibi-material\generated_data_by_deltaiot_simulation\under_drift_scenario'
# DATA_DIR = r'D:\projects\gheibi-material\generated_data_by_deltaiot_simulation\no_drift_scanerio'

LST_PACKET = []
LST_ENERGY = []
LST_DATA, _ = load_data(path=DATA_DIR, load_all=False, version='',
                      shuffle=False, fraction=1.0, test_size=0.2, return_train_test=False)
DATA = return_next_item(LST_DATA, normalize=False)
LST_DATAS = []
LST_PACKET = []
LST_ENERGY = []
LST_LATENCY = []
try:
    while(True):
        # LST_DATAS.append(next(DATA))
        data_ = next(DATA)
        LST_PACKET.append(data_['packetloss'].min())
        LST_ENERGY.append(data_['energyconsumption'].min())
        LST_LATENCY.append(data_['latency'].min())
except:
    # all_data = pd.concat(LST_DATAS)
    print('f')

    configurations = list(range(1, len(LST_LATENCY) + 1))
    # Create scatter plot
    # Create figure and axis objects
    fig, ax1 = plt.subplots()

    # Plot latency data
    color = 'tab:blue'
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Latency', color=color)
    ax1.scatter(configurations, LST_LATENCY, color=color, label='Latency')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for packet loss
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Packet Loss', color=color)
    ax2.plot(configurations, LST_PACKET, color=color, marker='o', linestyle='None', label='Packet Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and Grid
    plt.title('Latency and Packet Loss for Different Configurations')
    ax1.grid(True)

    # Show plot
    fig.tight_layout()
    plt.show()


    plt.scatter(LST_PACKET, LST_LATENCY, color='blue')
    plt.xlabel('Packet Loss')
    plt.ylabel('Latency')
    plt.title('Latency vs. Packet Loss')
    plt.grid(True)
    plt.show()
    

    # Create 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(LST_PACKET, LST_LATENCY, LST_ENERGY, c=LST_ENERGY, cmap='viridis')

    # Labels
    ax.set_xlabel('Packet Loss')
    ax.set_ylabel('Latency')
    ax.set_zlabel('Energy Consumption')
    plt.title('Latency vs. Packet Loss vs. Energy Consumption')

    # Color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Energy Consumption')

    # Show plot
    plt.show()


    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D


    # Meshgrid
    LST_ENERGY, LST_LATENCY = np.meshgrid(np.array(LST_ENERGY), np.array(LST_LATENCY))

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(LST_ENERGY, LST_LATENCY, np.array([LST_PACKET]), cmap='viridis')

    # Labels
    ax.set_xlabel('# Energy Consumption')
    ax.set_ylabel('# Latency')
    ax.set_zlabel('Packet loss')

    # Title
    plt.title('Latency as a Function of Counters and Splitters')

    # Show plot
    plt.show()

    print('f')

