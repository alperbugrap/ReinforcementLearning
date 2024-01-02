import gym
from stable_baselines3 import DQN
import os
import time
from snakeenv import SnekEnv


models_dir = f"models/{int(time.time())}"
logdir = f"logs/{int(time.time())}"
# models_dir = "models/1703426605"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = SnekEnv()
env.reset()


# model_path = f"{models_dir}/70000.zip"
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
# model = PPO.load(model_path, env=env)

TIMESTEPS = 10000
for i in range(1,100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()

# episodes = 500
# for ep in range(episodes):
#     obs = env.reset()
#     done = False
#     while not done:
#         action, _states = model.predict(obs)
#         obs, rewards, done, info, truncated = env.step(action)
#         print(rewards)
