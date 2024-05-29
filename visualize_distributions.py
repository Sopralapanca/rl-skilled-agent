from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, DummyVecEnv
import matplotlib.pyplot as plt
import argparse
import atari_py
import os
from stable_baselines3 import PPO
import torch as th

def episode_terminated(infos):
    return any(info.get('episode_done', False) for info in infos)

# parser = argparse.ArgumentParser()
# parser.add_argument("--env", help="Name of the environment to use i.e. Pong, Breakoout, etc.",
#                     type=str, required=True, choices=["Breakout", "Pong", "CartPole-v1",
#                                                       'Ms_Pacman', 'Seaquest', 'Qbert', 'Asteroids',
#                                                       'Enduro', 'Space_Invaders', 'Road_Runner', 'Beam_Rider'])
#
# args = parser.parse_args()

N_ENVS = 1
FRAME_STACK = 4
#ENV_NAME = args.env  # "Pong"
ENV_NAME = "Ms_Pacman"
model_path = "xbmyz15p" # ATTENZIONE CAMBIA MODELLO

# Create the environment
if ENV_NAME.lower() in atari_py.list_games():
    ENV_NAME = ENV_NAME.replace('_', '')
    ENV_NAME = ENV_NAME+"NoFrameskip-v4"
    vec_env = make_atari_env(ENV_NAME, n_envs=N_ENVS)
    action_meanings = vec_env.envs[0].unwrapped.get_action_meanings()
    action_space = vec_env.action_space

    vec_env = VecFrameStack(vec_env, n_stack=FRAME_STACK)
    vec_env = VecTransposeImage(vec_env)
else:
    raise NotImplementedError(ENV_NAME + " not implemented yet, try CartPole-v1 or one atari game")



obs = vec_env.reset()

if ENV_NAME.lower() in atari_py.list_games():
    vec_env.render("rgb_array")



model = PPO.load(f"./models/{model_path}/best_model.zip", device="cuda:1")

done = False
i = 0
# Define the min and max values for the x-axis
x_min, x_max = -1, 1  # Adjust these values based on your data range

while not done:
    action, _states = model.predict(obs) # returns a list of actions

    weights = model.policy.features_extractor.att_weights
    weights_label = list(weights.keys())

    tmp = model.policy.features_extractor.tmp
    embeddiings = model.policy.features_extractor.embeddings_values
    adapted_embeddings = model.policy.features_extractor.adapted_embeddings_values
    for j, (emb, adp, t) in enumerate(zip(embeddiings, adapted_embeddings, tmp)):
        tmp[j] = t.flatten().cpu().numpy()
        embeddiings[j] = emb.flatten().cpu().numpy()
        adapted_embeddings[j] = emb.flatten().cpu().numpy()

    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(16, 8))

    ax[0].hist(embeddiings, bins=20, label=["s1", "s2", "s3", "s4"], alpha=0.7, range=(x_min, x_max))
    ax[0].set_title("embeddings")
    ax[0].set_xlabel("values")
    ax[0].set_ylabel("frequency")
    ax[0].set_xlim(x_min, x_max)
    ax[0].legend()

    ax[1].hist(adapted_embeddings, bins=20, label=["s1", "s2", "s3", "s4"], alpha=0.7, range=(x_min, x_max))
    ax[1].set_title("adapted")
    ax[1].set_xlabel("values")
    ax[1].set_ylabel("frequency")
    ax[1].set_xlim(x_min, x_max)
    ax[1].legend()

    ax[2].hist(tmp, bins=20, label=["s1", "s2", "s3", "s4"], alpha=0.7, range=(x_min, x_max))
    ax[2].set_title("tmp")
    ax[2].set_xlabel("values")
    ax[2].set_ylabel("frequency")
    ax[2].set_xlim(x_min, x_max)
    ax[2].legend()

    plt.show()

    new_obs, rewards, dones, infos = vec_env.step(action)  # we need to pass an array of actions in step, one action for each environment
    obs = new_obs

    #done = episode_terminated(infos)
    done = dones[0]

    print("Step:", i)
    i = i + 1
    if i == 5:
       break


obs = vec_env.reset()
vec_env.close()