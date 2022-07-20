import gym
import numpy as np


def greedy(a):
    global e
    e = max(0.01, e - ((a + 1) / (n_iterations * 1000)))
    if np.random.rand() < e:
        return np.random.choice(action_space)
    else:
        return np.argmax(Q_Table[state])


if __name__ == '__main__':
    env = gym.make("FrozenLake-v1")

    obs = env.reset()

    state = 0
    action_space = [0, 1, 2, 3]
    Q_Table = np.full((16, 4), 0.0)
    n_iterations = 1000
    learning_rate = 0.01
    discount = 0.98
    rewards_tot = []

    e = 1.0

    for i in range(n_iterations):
        state = 0
        steps = 0
        while True:
            steps += 1
            action = greedy(i)
            obs, reward, done, _ = env.step(action)
            if done and reward != 1:
                reward = -1
            Q_Table[state, action] = ((1 - learning_rate) * Q_Table[state, action]) + learning_rate * (
                    reward + (discount * np.max(Q_Table[obs])))
            if done:
                print(i, ' reward=', reward, ' steps=', steps)
                obs = env.reset()
                break
            state = obs
            env.render()
    print(sum(rewards_tot), e, learning_rate)
    print(Q_Table)
    np.savetxt('D:\PyCharm Projects\Reinforcement Learning\Q Learning\Q_noepsilon3.csv', Q_Table, delimiter=',')
