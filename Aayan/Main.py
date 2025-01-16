import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = 'CartPole-v0'
env = gym.make(environment_name)

log_path = os.path.join('Training', 'Logs')
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=60000)

PPO_Path = os.path.join('Training', 'Saved Models', 'PPO_Model_Cartpole')
model.save(PPO_Path)
model = PPO.load(PPO_Path, env=env)

test_env = gym.make(environment_name, render_mode="human")

evaluate_policy(model, test_env, n_eval_episodes=10, render=True)

episodes = 5
for episode in range(1, episodes + 1):
    obs = test_env.reset()
    done = False
    score = 0

    while not done:
        test_env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = test_env.step(action)
        score += reward

    print('Episode: {} Score: {}'.format(episode, score))

env.close()
test_env.close()