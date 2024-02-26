import numpy as np
import matplotlib.pyplot as plt

# Number of arms (slot machines)
num_arms = 3

# True probabilities of success for each arm (unknown to the algorithm)
true_probs = np.array([0.3, 0.5, 0.8])

# Number of rounds or iterations
num_rounds = 1000

# Variables to store the number of successes and failures for each arm
successes = np.zeros(num_arms)
failures = np.zeros(num_arms)

# Variables to store cumulative rewards and regrets
cumulative_rewards = np.zeros(num_rounds)
regrets = np.zeros(num_rounds)

# Thompson Sampling algorithm
for round in range(num_rounds):
    # Sample from the posterior distribution for each arm
    sampled_probs = np.random.beta(successes + 1, failures + 1)
    
    # Choose the arm with the highest sampled probability
    chosen_arm = np.argmax(sampled_probs)
    
    # Simulate the reward for the chosen arm (0 or 1)
    reward = np.random.binomial(n=1, p=true_probs[chosen_arm])
    
    # Update successes and failures based on the observed reward
    if reward == 1:
        successes[chosen_arm] += 1
    else:
        failures[chosen_arm] += 1
    
    # Update cumulative rewards and regrets
    cumulative_rewards[round] = cumulative_rewards[round - 1] + reward if round > 0 else reward
    regrets[round] = np.max(true_probs) - true_probs[chosen_arm]

# Plot results
plt.plot(cumulative_rewards, label='Cumulative Rewards')
plt.plot(regrets, label='Regrets')
plt.xlabel('Round')
plt.ylabel('Value')
plt.legend()
plt.title('Thompson Sampling in Multi-Armed Bandit')
plt.show()
