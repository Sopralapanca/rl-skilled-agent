from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
import matplotlib.pyplot as plt
import argparse
import atari_py
import os
from stable_baselines3 import PPO
import moviepy.video.io.ImageSequenceClip

# parser = argparse.ArgumentParser()
# parser.add_argument("--env", help="Name of the environment to use i.e. Pong, Breakoout, etc.",
#                     type=str, required=True, choices=["Breakout", "Pong", "CartPole-v1",
#                                                       'Ms_Pacman', 'Seaquest', 'Qbert', 'Asteroids',
#                                                       'Enduro', 'Space_Invaders', 'Road_Runner', 'Beam_Rider'])
#
# args = parser.parse_args()

N_ENVS = 1
FRAME_STACK = 4
ENV_NAME = "Ms_Pacman"
model_path = "xbmyz15p"  # ATTENZIONE CAMBIA MODELLO
device = "cuda:0"

# Create the environment
if ENV_NAME.lower() in atari_py.list_games():
    ENV_NAME = ENV_NAME.replace('_', '')
    ENV_NAME = ENV_NAME + "NoFrameskip-v4"
    vec_env = make_atari_env(ENV_NAME, n_envs=N_ENVS)
    action_meanings = vec_env.envs[0].unwrapped.get_action_meanings()
    action_space = vec_env.action_space

    vec_env = VecFrameStack(vec_env, n_stack=FRAME_STACK)
    vec_env = VecTransposeImage(vec_env)
else:
    raise NotImplementedError(ENV_NAME + " not implemented yet, try CartPole-v1 or one atari game")


SAVE_DIR = "./distributions_data/" + ENV_NAME
# Create a directory data with subdirectory "breakout" using os to store the frames
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


obs = vec_env.reset()

if ENV_NAME.lower() in atari_py.list_games():
    vec_env.render("rgb_array")

model = PPO.load(f"./models/{model_path}/best_model.zip", device=device)


# Define the min and max values for the x-axis
#x_min, x_max = -10, 10  # Adjust these values based on your data range




bins = 30



done = False
i = 0
while not done:
    action, _states = model.predict(obs)  # returns a list of actions

    last_frame = obs[:, -1, :, :]

    weights = model.policy.features_extractor.att_weights
    weights_label = list(weights.keys())

    for index, name in enumerate(weights_label):
        if name == "state_rep_uns":
            weights_label[index] = "SR"
        if name == "obj_key_enc":
            weights_label[index] = "OKE"
        if name == "obj_key_key":
            weights_label[index] = "OKK"
        if name == "vid_obj_seg":
            weights_label[index] = "VOS"

    embeddings = model.policy.features_extractor.skills_embeddings
    spatial_emb = model.policy.features_extractor.spatial_adapters
    linear_emb = model.policy.features_extractor.linear_adapters
    weights = model.policy.features_extractor.att_weights
    values = [item for sublist in weights.values() for item in sublist]
    values = [v.item() for v in values]

    for j, (emb, sp, lin) in enumerate(zip(embeddings, spatial_emb, linear_emb)):
        embeddings[j] = emb.flatten().cpu().numpy()
        spatial_emb[j] = sp.flatten().cpu().numpy()
        linear_emb[j] = lin.flatten().cpu().numpy()


    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(40, 6))

    fig.suptitle(f"WSA Visualization - Step: {i}", fontsize=17)
    ax[0].imshow(last_frame[0], cmap='gray')
    ax[0].set_title('Last frame', fontsize=15)
    ax[0].axis('off')

    ax[1].bar(weights_label, values)
    ax[1].set_title('Attention Weights', fontsize=15)
    ax[1].set_ylim([0, 1])


    ax[2].hist(embeddings, bins=bins, label=weights_label)
    ax[2].set_title("embeddings")
    ax[2].set_xlabel("values")
    ax[2].set_ylabel("frequency")
    ax[2].legend()

    ax[3].hist(spatial_emb, bins=bins, label=weights_label)
    ax[3].set_title("spatial adapter")
    ax[3].set_xlabel("values")
    ax[3].set_ylabel("frequency")
    ax[3].legend()

    ax[4].hist(linear_emb, bins=bins, label=weights_label)
    ax[4].set_title("linear adapter")
    ax[4].set_xlabel("values")
    ax[4].set_ylabel("frequency")
    ax[4].legend()

    plt.savefig(f"{SAVE_DIR}/{i}.png")
    plt.close()

    new_obs, rewards, dones, infos = vec_env.step(
        action)  # we need to pass an array of actions in step, one action for each environment
    obs = new_obs

    #done = episode_terminated(infos)
    #done = dones[0]

    if infos[0].get("lives") == 0:
        break
    print("Step:", i)
    i = i + 1


obs = vec_env.reset()
vec_env.close()


# ------------------ VIDEO ------------------ #

fps = 5

image_files = [os.path.join(SAVE_DIR, img)
               for img in os.listdir(SAVE_DIR)
               if img.endswith(".png")]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile(SAVE_DIR + '/video.mp4')