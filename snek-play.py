import gym
from stable_baselines3 import PPO
import os
import time
from snakeenv import SnekEnv

models_dir = "models/1703551821"

env = SnekEnv()
env.reset()

model_path = f"{models_dir}/40000.zip"
model = PPO.load(model_path, env=env)

# model.timestamps = 400000

TIMESTEPS = 10000
for i in range(1, 100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()