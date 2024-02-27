import numpy as np
from scipy.stats import beta

class BayesianUCB:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.alpha = np.ones(num_arms)
        self.beta = np.ones(num_arms)
        self.trials = np.zeros(num_arms)

    def select_arm(self):
        # Calculate the Upper Confidence Bound for each arm
        ucb_values = np.zeros(self.num_arms)
        for arm in range(self.num_arms):
            if self.trials[arm] == 0:
                # Explore if an arm hasn't been pulled yet
                return arm
            else:
                ucb_values[arm] = beta.ppf(1 - 0.05, self.alpha[arm], self.beta[arm])

        # Select the arm with the highest UCB value
        chosen_arm = np.argmax(ucb_values)
        return chosen_arm

    def update(self, arm, reward):
        # Update the Beta distribution parameters based on the observed reward
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward
        self.trials[arm] += 1