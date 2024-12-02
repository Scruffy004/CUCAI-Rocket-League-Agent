# Reinforcement Learning with Stable Baselines3

This is a simple example of using Stable Baselines3, a library for reinforcement learning, to train an agent on the CartPole-v0 environment.

![CartPole](cartpole.gif)

## Dependencies
Make sure you have the following dependencies installed:
- stable-baselines3
- gym
- pyglet

You can install them using pip:
```
pip install stable-baselines3[extra]
pip install pyglet==1.5.27
```

## Load Environment
First, we import the necessary dependencies and create an instance of the CartPole-v0 environment:
```python
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = 'CartPole-v0'
env = gym.make(environment_name)
```

## Training
To train the agent, we initialize the PPO algorithm and pass in the environment. We then call the `learn` method to start the training process:
```python
log_path = os.path.join('Training', 'Logs')
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=20000)
```

## Training Logs
```
------------------------------------------
| time/                   |              |
|    fps                  | 581          |
|    iterations           | 10           |
|    time_elapsed         | 35           |
|    total_timesteps      | 20480        |
| train/                  |              |
|    approx_kl            | 0.0065331194 |
|    clip_fraction        | 0.0254       |
|    clip_range           | 0.2          |
|    entropy_loss         | -0.57        |
|    explained_variance   | 0.651        |
|    learning_rate        | 0.0003       |
|    loss                 | 7.36         |
|    n_updates            | 90           |
|    policy_gradient_loss | -0.0054      |
|    value_loss           | 23.8         |
------------------------------------------
```

## Save and Load the Model
You can save the trained model to a file and load it later for evaluation or further training:
```python
PPO_Path = os.path.join('Training', 'Saved Models', 'PPO_Model_Cartpole')
model.save(PPO_Path)
model = PPO.load(PPO_Path, env=env)
```

## Evaluation
To evaluate the performance of the trained agent, you can use the `evaluate_policy` function:
```python
evaluate_policy(model, env, n_eval_episodes=10, render=True)
```

## Testing the Model
You can test the trained model by running episodes and observing its behavior:
```python
episodes = 5
for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward

    print('Episode: {} Score: {}'.format(episode, score))

env.close()
```

## Viewing Training Logs
You can visualize the training progress using TensorBoard. First, specify the log directory and start TensorBoard:
```python
training_log_path = os.path.join(log_path, 'PPO_1')
!tensorboard --logdir={training_log_path}
```
Then, open `localhost:6006` in your browser to view the training logs.

## Conclusion
Reinforcement learning with Stable Baselines3 is a powerful tool for training agents in various environments. By following the steps in this example, you can train, save, and evaluate a reinforcement learning agent for the CartPole-v0 environment.