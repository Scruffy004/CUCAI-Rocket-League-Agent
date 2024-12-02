# Import all necessary libraries
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# function to draw the menu and take input
def drawMenu():
    print("Cartpole Main Menu")
    print("1. Train and run new agent")
    print("2. Load and run agent")
    print("3. exit")
    choice = int(input("Please choose an option: "))

    if choice == 1: # trains a new model with timestep input
        timesteps = int(input("Enter the number of timesteps: "))
        trainModel(timesteps)

    elif choice == 2: # runs a pre-existing model
        runModel()
    
    elif choice == 3: # exits
        exit()


# Function to train a model given timesteps
def trainModel(timesteps):
    # Create a gym environment (cartpole)
    environment_name = 'CartPole-v0'
    env = gym.make(environment_name)

    log_path = os.path.join('Training', 'Logs') # define path for logging
    env = DummyVecEnv([lambda: env]) 
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path) # define what model to use (PPO)

    model.learn(total_timesteps = timesteps) 

    PPO_Path = os.path.join('Training', 'Saved Models', 'PPO_Model_Cartpole') # define path to save agent
    model.save(PPO_Path) # save the agent to the path

    evaluate_policy(model, env, n_eval_episodes=10) # evaluate the policy

    drawMenu()


# Function for loading and running an existing agent
def runModel():
    # Create a gym environment (cartpole)
    environment_name = 'CartPole-v0'
    env = gym.make(environment_name, render_mode = "human")

    PPO_Path = os.path.join('Training', 'Saved Models', 'PPO_Model_Cartpole.zip') # define path to load agent
    
    # try/except to avoid non-existing agent issues
    try:
        model = PPO.load(PPO_Path, env=env) # load existing PPO model
        evaluate_policy(model, env, n_eval_episodes=10, render=True) # evaluate the policy

        testModel(model, env) # run the evaluated agent
    except:
        print("An error occured. You likely have not trained an agent yet.")


def testModel(model, env):
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


drawMenu()