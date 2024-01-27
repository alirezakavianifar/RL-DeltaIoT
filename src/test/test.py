import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.true_rewards = np.random.uniform(0, 1, num_arms)  # True reward probabilities for each arm
        self.estimated_rewards = np.zeros(num_arms)
        self.num_pulls = np.zeros(num_arms)
        self.timestep = 0

    def pull_arm(self, arm):
        # Simulate pulling the arm and receiving a reward (0 or 1)
        reward = np.random.binomial(1, self.true_rewards[arm])
        return reward

class UCB:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.total_rewards = np.zeros(num_arms)
        self.num_pulls = np.zeros(num_arms)
        self.timestep = 0

    def select_arm(self):
        if 0 in self.num_pulls:
            # Explore each arm at least once
            return np.where(self.num_pulls == 0)[0][0]
        else:
            # Use UCB formula to select arm
            exploration_term = np.sqrt(2 * np.log(self.timestep) / self.num_pulls)
            ucb_values = self.total_rewards / self.num_pulls + exploration_term
            return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        self.num_pulls[chosen_arm] += 1
        self.total_rewards[chosen_arm] += reward
        self.timestep += 1

# Number of arms in the bandit
num_arms = 3

# Create a multi-armed bandit and UCB algorithm instance
bandit = Bandit(num_arms)
ucb_algorithm = UCB(num_arms)

# Simulate bandit pulls
num_pulls = 1000
reward_history = []

for _ in range(num_pulls):
    chosen_arm = ucb_algorithm.select_arm()
    reward = bandit.pull_arm(chosen_arm)
    ucb_algorithm.update(chosen_arm, reward)
    reward_history.append(reward)

# Plot the cumulative reward over time
cumulative_reward = np.cumsum(reward_history)
timesteps = np.arange(1, num_pulls + 1)

plt.plot(timesteps, cumulative_reward, label='UCB')
plt.xlabel('Timesteps')
plt.ylabel('Cumulative Reward')
plt.title('Multi-Armed Bandit using UCB')
plt.legend()
plt.show()
















# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs
# from sklearn.decomposition import PCA
# from sklearn.datasets import load_iris
# from src.utility.utils import load_data, return_next_item
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from mpl_toolkits.mplot3d import Axes3D

# from src.utility.utils import pca_analysis, kmeans_analysis


# def plot_landscape(energy_consumption, packet_loss, latency):

# # Generate synthetic data
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D

#     # Create a 3D plot with a wireframe plot
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot the wireframe landscape
#     ax.plot_trisurf(energy_consumption, packet_loss, latency, cmap='viridis', edgecolor='black', linewidth=0.2)

#     # Add labels and title
#     ax.set_xlabel('Energy Consumption')
#     ax.set_ylabel('Packet Loss')
#     ax.set_zlabel('Latency')
#     ax.set_title('Non-Smooth Latency Landscape in IoT Setting')

#     # Show the plot
#     plt.show()


#     print('ff')


# def if_satisfied(row):
#     if row['energyconsumption'] < 12.9 and row['packetloss'] < 10 and row['latency'] < 5:
#         return 1
#     else:
#         return 0


# def getdata():
#     lst_X = []
#     lst_y = []
#     lst_perf = []
#     lst_train, lst_test = load_data(
#         r'D:\projects\RL-DelataIoT-fortest\RL-DeltaIoT\data\DeltaIoTv1\train')

#     df_handler = return_next_item(lst_train, normalize=False)

#     try:
#         while (True):
#             df = next(df_handler).drop('verification_times', axis=1)
#             scaler = StandardScaler()
#             df_X = pd.DataFrame(scaler.fit_transform(
#                 df['features'].values.tolist()))
#             X, y = pca_analysis(df_X)
#             kmeans_analysis(X, n_clusters=4)
#             lst_X.append(df_X)
#             # df['y'] = 1 if ((df['energyconsumption'] < 12.9) & (df['packetloss'] < 10) & (df['latency'] < 5)).all() else 0
#             df['y'] = df.apply(lambda x: if_satisfied(x), axis=1)
#             # df['y'] = 1 if ((df['energyconsumption'] < 12.9) & (df['packetloss'] < 10) & (df['latency'] < 5)) else 0
#             lst_y.append(df['y'])

#     except:
#         plot_landscape(df['energyconsumption'],
#                            df['packetloss'], df['latency'])
#         X = pd.concat(lst_X)
#         y = pd.concat(lst_y)

#     return X, y

if __name__ == '__main__':
    X, y = getdata()
#     X, y = pca_analysis(X, y)
#     kmeans_analysis(X, n_clusters=4)


