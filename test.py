import gym
import numpy as np

# Create FrozenLake environment
env = gym.make("FrozenLake-v1")

# Initialize observation
obs = env.reset()
done = False

# Counter to keep track of episodes
C = 0

# Load the pre-trained Q-table from the file
Q_Table = np.loadtxt('D:\PyCharm Projects\Reinforcement Learning\Q Learning\Q_noepsilon3.csv', delimiter=',')

# Number of episodes for testing
n_episodes = 50

# Maximum steps per episode
steps_per_episode = 1000

# List to store rewards for each episode
reward_tot = []

# Testing loop
for i in range(n_episodes):
    # Reset the environment for a new episode
    obs = env.reset()

    # Episode-specific loop
    for j in range(steps_per_episode):
        # Render the environment (optional)
        env.render()

        # Choose action based on the learned Q-table
        action = np.argmax(Q_Table[obs])

        # Take the chosen action and observe the new state and reward
        obs, reward, done, _ = env.step(action)

        # Break if the episode ends without reaching the goal
        if done and reward != 1:
            print(j)
            break

        # If the agent reaches the goal, print a success message and record the reward
        if reward == 1:
            print("Reward in episode", i)
            reward_tot.append(1)
            break

# Print the total number of episodes won out of the total tested episodes
print(sum(reward_tot), "episodes won out of ", n_episodes)
