import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, k=10):
        self.arms = k  # Number of arms
        self.rate = np.random.rand(k)  # True action values
    
    def play(self, arm):
        rate = self.rate[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0

class Agent:
    # epsilon-greedy agent
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.action_size = action_size
        self.Qs = np.zeros(action_size)
        self.Ns = np.zeros(action_size)
    
    def update(self, action, reward):
        self.Ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.Ns[action]

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            return np.argmax(self.Qs)


if __name__ == "__main__":
    np.random.seed(0)  # For reproducibility
    steps = 1000
    epsilon = 0.1

    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    total_rewards = []
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward
        total_rewards.append(total_reward)
        rates.append(total_reward / (step + 1))
    
    print(total_reward)
    print(rates[-1])

    # 累積報酬
    plt.figure()
    plt.plot(total_rewards)
    plt.xlabel('Steps')
    plt.ylabel('Total reward')
    plt.savefig("Total_reward.png", dpi=150, bbox_inches="tight")

    # レート
    plt.figure()
    plt.plot(rates)
    plt.xlabel('Steps')
    plt.ylabel('Rates')
    plt.savefig("rates.png", dpi=150, bbox_inches="tight")
