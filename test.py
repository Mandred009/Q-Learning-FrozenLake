import gym
import numpy as np

env = gym.make("FrozenLake-v1")

obs = env.reset()
done = False
C = 0

Q_Table = np.loadtxt('D:\PyCharm Projects\Reinforcement Learning\Q Learning\Q_noepsilon3.csv', delimiter=',')

n_episodes = 50
steps_per_episode = 1000

reward_tot = []
for i in range(n_episodes):
    obs = env.reset()
    for j in range(steps_per_episode):
        env.render()
        action = np.argmax(Q_Table[obs])
        obs, reward, done, _ = env.step(action)
        if done and reward!=1:
            print(j)
            break
        if reward == 1:
            print("reward",i)
            reward_tot.append(1)
            break

print(sum(reward_tot), "episodes won out of ", n_episodes)
