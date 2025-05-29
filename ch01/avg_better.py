import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
avgs = []

Q = np.random.rand()  # Initialize Q with a random value
avgs.append(Q)

for i in range(1, 10000000):
    reward = np.random.rand()
    Q = Q + (reward - Q) / i
    avgs.append(Q)

# Plot the average reward
plt.plot(avgs, label='Average Reward')
plt.xlabel('Number of Trials')
plt.ylabel('Average Reward')
plt.title('Average Reward Over Time')
plt.legend()
plt.savefig("output.png", dpi=150, bbox_inches="tight")
print("Average reward after 10000000 trials:", avgs[-1])
