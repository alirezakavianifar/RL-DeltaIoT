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


conf_num = 1
try:
    while(True):
        # LST_DATAS.append(next(DATA))
        data = next(DATA)
        LST_PACKET.append(data.iloc[conf_num]['packetloss'])
        LST_ENERGY.append(data.iloc[conf_num]['energyconsumption'])
        LST_LATENCY.append(data.iloc[conf_num]['latency'])
except:
    
    configurations = list(range(1, len(LST_LATENCY) + 1))

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

    
    plt.scatter(configurations, LST_LATENCY, color='blue')
    plt.xlabel('environments')
    plt.ylabel('Latency')
    plt.title('environments vs. Latency')
    plt.grid(True)
    plt.show()


    plt.scatter(configurations, LST_PACKET, color='blue')
    plt.xlabel('environments')
    plt.ylabel('Packet Loss')
    plt.title('environments vs. Packet Loss')
    plt.grid(True)
    plt.show()

    plt.scatter(configurations, LST_ENERGY, color='blue')
    plt.xlabel('environments')
    plt.ylabel('Energy Consumption')
    plt.title('environments vs. Packet Loss')
    plt.grid(True)
    plt.show()

    plt.plot(configurations, LST_PACKET, color='blue')
    plt.xlabel('environments')
    plt.ylabel('Packet Loss')
    plt.title('environments vs. Packet Loss')
    plt.grid(True)
    plt.show()

    print('f')

