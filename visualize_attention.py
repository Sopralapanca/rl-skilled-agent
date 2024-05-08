from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
import matplotlib.pyplot as plt
import argparse
import atari_py

from stable_baselines3 import PPO

parser = argparse.ArgumentParser()
parser.add_argument("--env", help="Name of the environment to use i.e. Pong, Breakoout, etc.",
                    type=str, required=True, choices=["Breakout", "Pong", "CartPole-v1",
                                                      'Ms_Pacman', 'Seaquest', 'Qbert', 'Asteroids',
                                                      'Enduro', 'Space_Invaders', 'Road_Runner', 'Beam_Rider'])

args = parser.parse_args()

N_ENVS = 1
NUM_EPS = 1000
FRAME_STACK = 4
ENV_NAME = args.env  # "Pong"

# Create the environment
if ENV_NAME.lower() in atari_py.list_games():
    ENV_NAME = ENV_NAME.replace('_', '')
    ENV_NAME = ENV_NAME+"NoFrameskip-v4"
    vec_env = make_atari_env(ENV_NAME, n_envs=N_ENVS)
    vec_env = VecFrameStack(vec_env, n_stack=FRAME_STACK)
    vec_env = VecTransposeImage(vec_env)
else:
    raise NotImplementedError(ENV_NAME + " not implemented yet, try CartPole-v1 or one atari game")

obs = vec_env.reset()

if ENV_NAME.lower() in atari_py.list_games():
    vec_env.render("rgb_array")

model_path = "y7gry5qu" #INSERIRE
model = PPO.load(f"./models/{model_path}/best_model.zip")


for i in range(1000):
    action, _states = model.predict(obs) # returns a list of actions

    # weights = model.policy.features_extractor.att_weights.cpu().numpy()
    # print(weights)

    last_frame = obs[:, -1, :, :]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(last_frame[0], cmap='gray')
    ax[1].imshow(last_frame[0], cmap='gray')
    plt.show()
    new_obs, rewards, dones, infos = vec_env.step(action)  # we need to pass an array of actions in step, one action for each environment
    obs = new_obs
    break
obs = vec_env.reset()
vec_env.close()