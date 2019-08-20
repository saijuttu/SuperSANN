import tensorflow as tf
import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import DQN

env = gym.make('MountainCar-v0')
env.reset()
env._max_episodes = 3000

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(24, input_shape=(4,), activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='linear'))
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=.001))

model = tf.keras.models.load_model("traditional.hd5")

episode = []
memory = []

agent = DQN.Agent(.95, .5, .999, model, env)
print(env.observation_space.shape[0])
agent.train(100, 10, 64)
agent.playGame()

model.save("model.hd5")

env.close()
