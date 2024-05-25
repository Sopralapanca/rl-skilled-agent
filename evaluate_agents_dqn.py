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
from stable_baselines3 import DQN
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
device = "cuda:1"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # ignore tensorflow warnings about CPU
n_seeds = 5
seeds = [np.random.randint(0, 100000) for i in range(n_seeds)]
eval_episodes = 20

results_dir = "./results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

path = results_dir + "/eval_results_dqn2.csv"
if os.path.isfile(path):
    df = pd.read_csv(path, index_col=0)

else:
    df = pd.DataFrame(columns=["env", "agent", "seed", "mean_reward", "std_reward"])

d = {"Ms_Pacman-3": {"DQN": ["9fcsfg0q", "qait7iqd", "ux3dq649", "d22cq9mj"],

                     "wsharing_attention_ext": ["2kt7afqj", "wpvnmnep", "qz3qi9rd", "zc8fhqcc"]
                     },

     "Breakout-Expert": {"DQN": ["5a66lrbw", "b75w1mj5", "ezg1j2ni", "sggf6fi1"],
                  "wsharing_attention_ext": ["ayaor062", "6lgipksw", "ssg08m8t", "dhkwd3sz"]
                  },
     }

for seed in seeds:
    for env in d.keys():

        if os.path.isfile(path):
            df = pd.read_csv(path, index_col=0)
        else:
            df = pd.DataFrame(columns=["env", "agent", "seed", "mean_reward", "std_reward"])

        env_name = env.split("-")[0]
        agents = d[env]

        if not os.path.exists(results_dir + "/" + env_name):
            os.makedirs(results_dir + "/" + env_name)

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

            models = agents[agent]

            if agent != "DQN":
                if "Expert" in env:
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

                config["f_ext_name"] = agent

                if agent == "wsharing_attention_ext":
                    config["f_ext_class"] = WeightSharingAttentionExtractor
                    config["game"] = vec_env_name
                    features_dim = 256

                f_ext_kwargs = config["f_ext_kwargs"]

                if agent == "wsharing_attention_ext":
                    f_ext_kwargs["game"] = vec_env_name
                    f_ext_kwargs["expert"] = True if "Expert" in env else False

                f_ext_kwargs["skills"] = skills
                f_ext_kwargs["features_dim"] = features_dim

                policy_kwargs = dict(
                    features_extractor_class=config["f_ext_class"],
                    features_extractor_kwargs=f_ext_kwargs,
                )
                custom_objects = {"policy_kwargs": policy_kwargs}

                for m in models:
                    load_path = f"./models/{m}/best_model.zip"
                    model = DQN.load(path=load_path, env=vec_env, device=device,
                                     custom_objects=custom_objects)  # don't need to pass policy_kwargs
                    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=eval_episodes)
                    print(f"Env:{env} Agent:{agent} Seed:{seed} Mean reward:{mean_reward:.2f} +/- {std_reward:.2f}")
                    df.loc[len(df.index)] = [env, agent, seed, mean_reward, std_reward]

            else:
                for m in models:
                    load_path = f"./models/{m}/best_model.zip"
                    model = DQN.load(path=load_path, env=vec_env, device=device)  # don't need to pass policy_kwargs

                    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=eval_episodes)
                    print(f"Env:{env} Agent:{agent} Seed:{seed} Mean reward:{mean_reward:.2f} +/- {std_reward:.2f}")
                    df.loc[len(df.index)] = [env, agent, seed, mean_reward, std_reward]

        df.to_csv(path)
