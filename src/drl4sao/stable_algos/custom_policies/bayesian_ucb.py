import numpy as np
from scipy.stats import beta


class BayesianUCB:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.alpha = np.ones(num_arms)
        self.beta = np.ones(num_arms)
        self.trials = np.zeros(num_arms)

    def get_ucb_values(self):
        # Calculate the Upper Confidence Bound for each arm
        ucb_values = np.zeros(self.num_arms)
        
        ucb_values = beta.ppf(1 - 0.05, self.alpha, self.beta)
        ucb_values[self.trials == 0] = np.inf

        # Select the arm with the highest UCB value

        return ucb_values

    def select_arm(self):
        # Select the arm with the highest UCB value
        ucb_values = self.get_ucb_values()
        chosen_arm = np.argmax(ucb_values)
        return chosen_arm

    def update(self, arm, reward):
        # Update the Beta distribution parameters based on the observed reward
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward
        self.trials[arm] += 1
