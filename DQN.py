import tensorflow as tf
import gym
import numpy as np
import random
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, y, eps, decay_rate, model, env):
        self.y = y
        self.eps = eps
        self.decay_rate = decay_rate
        self.model = model
        self.env = env
        self.memory = []
        self.episode = []
        self.observation_length = env.observation_space.shape[0]

    def playGame(self, games=20, time_steps=1000):
        for i in range(games):
            s = self.env.reset()
            for t in range(time_steps):
                self.env.render()
                s = np.reshape(s, [1, self.observation_length])
                action = np.argmax(self.model.predict(s))
                s, reward, done, info = self.env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break

    def train(self, num_games, num_random_games, mini_batch):
        self.random_games(num_random_games)
        r_avg_list = []
        for gidx in range(num_games):
            s = self.env.reset()
            s = np.reshape(s, [1, self.observation_length])
            self.eps *= self.decay_rate
            done = False
            r_sum = 0
            i = 0
            while not done:
                self.env.render()
                i += 1
                if np.random.random() < self.eps:
                    a = self.env.action_space.sample()
                else:
                    a = np.argmax(self.model.predict(s))
                new_s, r, done, _ = self.env.step(a)
                new_s = np.reshape(new_s, [1, self.observation_length])
                self.memory.append((s, a, r, new_s, done))
                if len(self.memory) > mini_batch:
                    for state, action, reward, next_state, done in random.sample(self.memory, mini_batch):
                        target = reward
                        if not done:
                            target = reward + self.y * np.max(self.model.predict(next_state)[0])
                            target_f = self.model.predict(state)
                            target_f[0][action] = target
                            self.model.fit(state, target_f, epochs=1, verbose=0)
                s = new_s
                r_sum += r
            print("Episode {} of {} Score: {}".format(gidx + 1, num_games, i))
            r_avg_list.append(i)
            self.episode.append(gidx + 1)
        plt.plot(self.episode, r_avg_list)
        plt.show()

    def random_games(self, num_random_games):
        for gidx in range(num_random_games):
            s = self.env.reset()
            s = np.reshape(s, [1, self.observation_length])
            done = False
            while not done:
                a = self.env.action_space.sample()
                new_s, r, done, _ = self.env.step(a)
                new_s = np.reshape(new_s, [1, self.observation_length])
                self.memory.append((s, a, r, new_s, done))
                s = new_s