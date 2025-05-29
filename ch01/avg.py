import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

rewards = []
avgs = []
for i in range(10000):
    # Generate 100 random numbers
    reward = np.random.rand()
    rewards.append(reward)
# Calculate the average of the rewards
    average_reward = np.mean(rewards)
    avgs.append(average_reward)

# Plot the average reward
plt.plot(avgs, label='Average Reward')
plt.xlabel('Number of Trials')
plt.ylabel('Average Reward')
plt.title('Average Reward Over Time')
plt.legend()
plt.savefig("output.png", dpi=150, bbox_inches="tight")
print("Average reward after 10000 trials:", avgs[-1])