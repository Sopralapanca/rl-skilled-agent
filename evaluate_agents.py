# general imports
import yaml
import numpy as np
import random
import os
import torch as th

# testing imports
from skill_models import *
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import tensorflow as tf
from stable_baselines3.common.vec_env import VecVideoRecorder

# utility imports
from utils.args import parse_args
import pandas as pd

from skill_models import *
from feature_extractors import LinearConcatExtractor, FixedLinearConcatExtractor, \
    CNNConcatExtractor, CombineExtractor, \
    DotProductAttentionExtractor, WeightSharingAttentionExtractor, \
    ReservoirConcatExtractor
import argparse
# ---------------------------------- MAIN ----------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--env", help="Name of the environment to use i.e. Pong",
                    type=str, required=True, choices=['Pong', 'Ms_Pacman', 'Breakout'])
parser.add_argument("--device", help="Integer number of a device to use (0, 1, 2, 3), or cpu",
                    type=str, default="cpu", required=False, choices=["cpu", "0", "1", "2", "3"])
args = parser.parse_args()

env_name = args.env
device = f"cuda:{args.device}"


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # ignore tensorflow warnings about CPU
n_seeds = 10
seeds = [np.random.randint(0, 100000) for i in range(n_seeds)]
eval_episodes = 100

results_dir = "./results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

path = results_dir + "/eval_results.csv"
if os.path.isfile(path):
    df = pd.read_csv(path, index_col=0)

else:
    df = pd.DataFrame(columns=["env", "agent", "seed", "mean_reward", "std_reward"])

if env_name == "Pong":
    agents = {"PPO": "k24cn512",
              "wsharing_attention_ext": "vwll3bv1",
              "reservoir_concat_ext": "025abyrl",
              "cnn_concat_ext": "yyt0d5xr",
              }
elif env_name == "Ms_Pacman":
    agents = {"PPO": "8l5cbixu",
              "wsharing_attention_ext": "xbmyz15p",
              "reservoir_concat_ext": "88rmd7an",
              "cnn_concat_ext": "0vm9cdpz",
              }

if not os.path.exists(results_dir + "/" + env_name):
    os.makedirs(results_dir + "/" + env_name)

for seed in seeds:
    for agent in agents.keys():
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if "_" in env_name:
            vec_env_name = env_name.replace("_", "")
        else:
            vec_env_name = env_name

        vec_env = make_atari_env(f"{vec_env_name}NoFrameskip-v4", n_envs=1, seed=seed)
        vec_env = VecFrameStack(vec_env, n_stack=4)
        vec_env = VecTransposeImage(vec_env)
        vec_env = VecVideoRecorder(vec_env,
                                   f"{results_dir}/{env_name}/",
                                   record_video_trigger=lambda x: x % 2000 == 0,
                                   video_length=10000,
                                   name_prefix=f"{agent}_{seed}"
                                   )

        load_path = f"./models/{agents[agent]}/best_model.zip"

        if agent != "PPO":
            if env_name == "Breakout":
                expert = True
            else:
                expert = False

            skills = []
            skills.append(get_state_rep_uns(vec_env_name, device, expert=expert))
            skills.append(get_object_keypoints_encoder(vec_env_name, device, load_only_model=True, expert=expert))
            skills.append(get_object_keypoints_keynet(vec_env_name, device, load_only_model=True, expert=expert))
            skills.append(get_video_object_segmentation(vec_env_name, device, load_only_model=True, expert=expert))

            sample_obs = vec_env.observation_space.sample()
            sample_obs = torch.tensor(sample_obs).to(device)
            sample_obs = sample_obs.unsqueeze(0)

            with open(f'configs/{env_name.lower()}.yaml', 'r') as file:
                config = yaml.safe_load(file)["config"]

            config["f_ext_kwargs"]["device"] = device

            if env_name == "Breakout":
                config["net_arch_pi"] = [1024, 512, 256]
                config["net_arch_vf"] = [1024, 512, 256]
            else:
                config["net_arch_pi"] = [256]
                config["net_arch_vf"] = [256]

            config["f_ext_name"] = agent
            if agent == "wsharing_attention_ext":
                config["f_ext_class"] = WeightSharingAttentionExtractor
                config["game"] = vec_env_name
                if env_name == "Pong":
                    features_dim = 1024
                if env_name == "Ms_Pacman":
                    features_dim = 256

            elif agent == "reservoir_concat_ext":
                config["f_ext_class"] = ReservoirConcatExtractor
                ext = LinearConcatExtractor(vec_env.observation_space, skills=skills, device=device)
                input_features_dim = ext.get_dimension(sample_obs)
                features_dim = 1024

            elif agent == "cnn_concat_ext":
                ext = CNNConcatExtractor(vec_env.observation_space, skills=skills, device=device)
                features_dim = ext.get_dimension(sample_obs)
                config["f_ext_class"] = CNNConcatExtractor

            f_ext_kwargs = config["f_ext_kwargs"]

            if agent == "wsharing_attention_ext":
                f_ext_kwargs["game"] = vec_env_name
                f_ext_kwargs["expert"] = False if env_name != "Breakout" else True

            elif agent == "reservoir_concat_ext":
                f_ext_kwargs["input_features_dim"] = input_features_dim

            elif agent == "cnn_concat_ext":
                f_ext_kwargs["num_conv_layers"] = 2

            f_ext_kwargs["skills"] = skills
            f_ext_kwargs["features_dim"] = features_dim

            if env_name == "Breakout":
                policy_kwargs = dict(
                    features_extractor_class=config["f_ext_class"],
                    features_extractor_kwargs=f_ext_kwargs,
                    net_arch={
                        'pi': config["net_arch_pi"],
                        'vf': config["net_arch_vf"]
                    },
                    activation_fn=th.nn.ReLU,
                )
            else:
                policy_kwargs = dict(
                    features_extractor_class=config["f_ext_class"],
                    features_extractor_kwargs=f_ext_kwargs,
                    net_arch={
                        'pi': config["net_arch_pi"],
                        'vf': config["net_arch_vf"]
                    }
                )
            custom_objects = {"policy_kwargs": policy_kwargs}
            model = PPO.load(path=load_path, env=vec_env, device=device,
                             custom_objects=custom_objects)  # don't need to pass policy_kwargs

        else:
            model = PPO.load(path=load_path, env=vec_env, device=device)  # don't need to pass policy_kwargs

        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=eval_episodes)
        print(f"Agent:{agent} Seed: {seed} Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        df.loc[len(df.index)] = [env_name, agent, seed, mean_reward, std_reward]

df.to_csv(path)
