import gym
import numpy as np

def greedy(a):
    global e
    # Exploration-exploitation trade-off: Reduce exploration (epsilon) over time
    e = max(0.01, e - ((a + 1) / (n_iterations * 1000)))
    # Exploration: Random action with probability epsilon
    if np.random.rand() < e:
        return np.random.choice(action_space)
    # Exploitation: Choose the action with the highest Q-value
    else:
        return np.argmax(Q_Table[state])

if __name__ == '__main__':
    # Create the FrozenLake environment
    env = gym.make("FrozenLake-v1")

    # Initialize environment
    obs = env.reset()

    # Initial state
    state = 0

    # Action space (0: left, 1: down, 2: right, 3: up)
    action_space = [0, 1, 2, 3]

    # Q-table initialization
    Q_Table = np.full((16, 4), 0.0)

    # Number of training iterations
    n_iterations = 1000

    # Learning rate and discount factor
    learning_rate = 0.01
    discount = 0.98

    # List to store total rewards at each iteration
    rewards_tot = []

    # Initial exploration rate
    e = 1.0

    # Training loop
    for i in range(n_iterations):
        # Reset the environment and initialize state
        state = 0
        steps = 0

        # Episode loop
        while True:
            steps += 1

            # Choose action using epsilon-greedy strategy
            action = greedy(i)

            # Take the chosen action and observe the new state and reward
            obs, reward, done, _ = env.step(action)

            # If the episode ends and the agent didn't reach the goal, penalize with a reward of -1
            if done and reward != 1:
                reward = -1

            # Q-table update using the Q-learning formula
            Q_Table[state, action] = ((1 - learning_rate) * Q_Table[state, action]) + learning_rate * (
                    reward + (discount * np.max(Q_Table[obs])))

            # If the episode is done, print information and reset the environment
            if done:
                print(i, ' reward=', reward, ' steps=', steps)
                obs = env.reset()
                break

            # Update the current state
            state = obs

            # Render the environment (optional)

    # Print the total rewards, final epsilon value, and the learned Q-table
    print(sum(rewards_tot), e, learning_rate)
    print(Q_Table)

    # Save the learned Q-table to a CSV file
    np.savetxt('D:\PyCharm Projects\Reinforcement Learning\Q Learning\Q_noepsilon3.csv', Q_Table, delimiter=',')
