from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, DummyVecEnv
import matplotlib.pyplot as plt
import argparse
import atari_py
import os
from stable_baselines3 import PPO
import os
import moviepy.video.io.ImageSequenceClip
from utils.load_custom_policykwargs import load_policy_kwargs


def episode_terminated(infos):
    return any(info.get('episode_done', False) for info in infos)


parser = argparse.ArgumentParser()
parser.add_argument("--env", help="Name of the environment to use i.e. Pong, Breakoout, etc.",
                    type=str, required=True, choices=["Breakout", "Pong", "CartPole-v1",
                                                      'Ms_Pacman', 'Seaquest', 'Qbert', 'Asteroids',
                                                      'Enduro', 'Space_Invaders', 'Road_Runner', 'Beam_Rider'])

args = parser.parse_args()

N_ENVS = 1
FRAME_STACK = 4
ENV_NAME = args.env  # "Pong"
model_path = "3mo21eg2"  # change this to the model you want to visualize
device = "cuda:2"
info = "_entropy0-001"
feature_dim = 1024 if "Pong" in ENV_NAME else 256
net_arch = [256]
custom_object = load_policy_kwargs(expert=False, device=device, env=ENV_NAME,
                                   net_arch=net_arch, agent="wsharing_attention_ext",
                                   features_dim=feature_dim, num_conv_layers=0)

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

SAVE_DIR = "./attention_data/" + ENV_NAME + info
# Create a directory data with subdirectory "breakout" using os to store the frames
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

obs = vec_env.reset()

if ENV_NAME.lower() in atari_py.list_games():
    vec_env.render("rgb_array")

model = PPO.load(f"./models/{model_path}/best_model.zip", device=device, custom_objects=custom_object)
offset = 1 / len(action_meanings)
done = False
i = 0
while not done:
    action, _states = model.predict(obs)  # returns a list of actions

    weights = model.policy.features_extractor.att_weights
    weights_label = list(weights.keys())

    for index, l in enumerate(weights_label):
        if l == "state_rep_uns":
            weights_label[index] = "SR"
        if l == "obj_key_enc":
            weights_label[index] = "OKE"
        if l == "obj_key_key":
            weights_label[index] = "OKK"
        if l == "vid_obj_seg":
            weights_label[index] = "VOS"

    values = [item for sublist in weights.values() for item in sublist]
    values = [v.item() for v in values]

    last_frame = obs[:, -1, :, :]

    fig, ax = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1, 0.2]})
    fig.suptitle(f"WSA Visualization - Step: {i}", fontsize=17)
    ax[0].imshow(last_frame[0], cmap='gray')
    ax[0].set_title('Last frame', fontsize=15)
    ax[0].axis('off')

    ax[1].bar(weights_label, values)
    ax[1].set_title('Attention Weights', fontsize=15)
    ax[1].set_ylim([0, 1])
    # Plot the list of names in a new subplot on the right
    ax[2].set_title('Chosen Action', fontsize=15)  # Title for the subplot

    vertical_offset = 0.1
    horizontal_offset = 0.5
    for j, a in enumerate(action_meanings):
        if action[0] == j:
            ax[2].text(horizontal_offset, 1 - vertical_offset, a, fontsize=15, ha='center', color="blue")
        else:
            ax[2].text(horizontal_offset, 1 - vertical_offset, a, fontsize=14, ha='center', color="gray")
        vertical_offset += offset  # Adjust this value to control vertical spacing

    ax[2].axis('off')

    #save the figure
    plt.savefig(f"{SAVE_DIR}/{i}.png")
    plt.close()

    new_obs, rewards, dones, infos = vec_env.step(
        action)  # we need to pass an array of actions in step, one action for each environment
    obs = new_obs

    if "Pong" in ENV_NAME:
        done = dones[0]
        print(f"Step:{i}")
    else:
        print(f"Step:{i} lives:{infos[0].get('lives')}")
        if infos[0].get("lives") == 0:
            break

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
