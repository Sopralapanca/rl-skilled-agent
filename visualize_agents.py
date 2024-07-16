from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, DummyVecEnv, VecVideoRecorder
import matplotlib.pyplot as plt
import argparse
import atari_py
import os
from stable_baselines3 import PPO
import os
import moviepy.video.io.ImageSequenceClip
from utils.load_custom_policykwargs import load_policy_kwargs

parser = argparse.ArgumentParser()
parser.add_argument("--env", help="Name of the environment to use i.e. Pong, Breakoout, etc.",
                    type=str, required=True, choices=["Breakout", "Pong", "CartPole-v1",
                                                      'Ms_Pacman', 'Seaquest', 'Qbert', 'Asteroids',
                                                      'Enduro', 'Space_Invaders', 'Road_Runner', 'Beam_Rider'])

args = parser.parse_args()

N_ENVS = 1
FRAME_STACK = 4
ENV_NAME = args.env  # "Pong"
model_path = "ba5ow0zz"  # change this to the model you want to visualize
device = "cpu"
feature_dim = 1024 if "Pong" in ENV_NAME else 256
net_arch = [1024, 512, 256] if "Breakout" in ENV_NAME else [256]

expert = True if "Breakout" in ENV_NAME else False
custom_object = load_policy_kwargs(expert=expert, device=device, env=ENV_NAME,
                                   net_arch=net_arch, agent="wsharing_attention_ext",
                                   features_dim=feature_dim, num_conv_layers=0)

# Create the environment
if ENV_NAME.lower() in atari_py.list_games():
    ENV_NAME = ENV_NAME.replace('_', '')
    ENV_NAME = ENV_NAME + "NoFrameskip-v4"
    vec_env = make_atari_env(ENV_NAME, n_envs=N_ENVS)
    #vec_env = make_vec_env(ENV_NAME, n_envs=N_ENVS) #can't use this as skills are trained for 84x84 pixels images

    vec_env = VecFrameStack(vec_env, n_stack=FRAME_STACK)
    vec_env = VecTransposeImage(vec_env)
else:
    raise NotImplementedError(ENV_NAME + " not implemented yet, try CartPole-v1 or one atari game")

SAVE_DIR = "./videos/" + ENV_NAME
# Create a directory data with subdirectory "breakout" using os to store the frames
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# if ENV_NAME.lower() in atari_py.list_games():
#     vec_env.render("rgb_array")

# Record the video starting at the first step
video_length = 10000
vec_env = VecVideoRecorder(vec_env, SAVE_DIR,
                           record_video_trigger=lambda x: x == 0, video_length=video_length,
                           name_prefix=f"wsa-{ENV_NAME}")

model = PPO.load(f"./models/{model_path}/best_model.zip", device=device, custom_objects=custom_object)

obs = vec_env.reset()
for _ in range(video_length + 1):
    action, _states = model.predict(obs)
    obs, _, _, _ = vec_env.step(action)

vec_env.close()
