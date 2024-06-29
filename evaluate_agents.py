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
device = "cuda:3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # ignore tensorflow warnings about CPU
#n_seeds = 5
#seeds = [np.random.randint(0, 100000) for i in range(n_seeds)]
seeds = [47695, 32558, 94088, 71782, 66638]
eval_episodes = 20

results_dir = "./results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

path = results_dir + "/eval_results4.csv"
if os.path.isfile(path):
    df = pd.read_csv(path, index_col=0)

else:
    df = pd.DataFrame(columns=["env", "agent", "seed", "mean_reward", "std_reward"])



# TODO: change the code below using utilis/load_custom_policykwargs.py instead
d = {
    # "Pong":
    #      {"PPO": "k24cn512",
    #       "wsharing_attention_ext": "vwll3bv1",
    #       "reservoir_concat_ext": "025abyrl",
    #       "cnn_concat_ext": "yyt0d5xr",
    #       },
    # "Ms_Pacman": {"PPO": "8l5cbixu",
    #            "wsharing_attention_ext": "xbmyz15p",
    #            "reservoir_concat_ext": "88rmd7an",
    #            "cnn_concat_ext": "0vm9cdpz",
    #            },

    "Breakout": {"PPO": ["01v1cwww", "5jb6tuoo", "16jzci1z"],  #"ycp3r13u"
                 "wsharing_attention_ext": ["jv6j5upf", "2zw7ox2c", "dbjc29fy"],  #"ckd8d160",
                 # "fixed_lin_concat_ext": "gy9a4wow",
                 # "cnn_concat_ext": "6qqnn3ce"
                 },

    "Breakout-Policy": {
        "wsharing_attention_ext": ["trl9zgvt", "1l26vwat", "v4pnpue5"],  #"j934tseo"
        # "fixed_lin_concat_ext": "g1bfh8y9", #["g1bfh8y9", "cuda:3"],
        # "cnn_concat_ext": "zl8boshh" #["zl8boshh", "cuda:2"],
    },

    "Breakout-Expert": {
        "wsharing_attention_ext": ["34i1meeg", "h763hmxr", "645m7wq9"],  #"12n3bzj9"
        # "fixed_lin_concat_ext": "mdmh29il", #["mdmh29il", "cuda:3"],
        # "cnn_concat_ext": "oh2n2o7g" #["oh2n2o7g", "cuda:1"],
    },

    "Breakout-Expert_Policy": {
        "wsharing_attention_ext": ["e5x2bp6w", "c6y9rvdu", "b59ggnda"]  #"ba5ow0zz"
        # "fixed_lin_concat_ext": "npa2880u", #["npa2880u", "cuda:1"],
        # "cnn_concat_ext": "0mcyd522", #["0mcyd522", "cuda:1"],
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
            if agent != "PPO":
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

                if "Policy" in env:
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
                    if env_name == "Ms_Pacman" or env_name == "Breakout":
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

                elif agent == "fixed_lin_concat_ext":
                    config["f_ext_class"] = FixedLinearConcatExtractor
                    ext = FixedLinearConcatExtractor(observation_space=vec_env.observation_space, skills=skills,
                                                     device=device, fixed_dim=512)
                    features_dim = ext.get_dimension(sample_obs)

                f_ext_kwargs = config["f_ext_kwargs"]

                if agent == "wsharing_attention_ext":
                    f_ext_kwargs["game"] = vec_env_name
                    f_ext_kwargs["expert"] = True if "Expert" in env else False

                elif agent == "reservoir_concat_ext":
                    f_ext_kwargs["input_features_dim"] = input_features_dim

                elif agent == "cnn_concat_ext":
                    if env_name == "Breakout":
                        f_ext_kwargs["num_conv_layers"] = 3
                    else:
                        f_ext_kwargs["num_conv_layers"] = 2

                elif agent == "fixed_lin_concat_ext":
                    f_ext_kwargs["fixed_dim"] = 512

                f_ext_kwargs["skills"] = skills
                f_ext_kwargs["features_dim"] = features_dim

                if "Policy" in env:
                    policy_kwargs = dict(
                        features_extractor_class=config["f_ext_class"],
                        features_extractor_kwargs=f_ext_kwargs,
                        net_arch={
                            'pi': config["net_arch_pi"],
                            'vf': config["net_arch_vf"],
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

                for m in models:
                    load_path = f"./models/{m}/best_model.zip"
                    model = PPO.load(path=load_path, env=vec_env, device=device,
                                     custom_objects=custom_objects)  # don't need to pass policy_kwargs
                    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=eval_episodes)
                    print(f"Env:{env} Agent:{agent} Seed:{seed} Mean reward:{mean_reward:.2f} +/- {std_reward:.2f}")
                    df.loc[len(df.index)] = [env, agent, seed, mean_reward, std_reward]

            else:
                for m in models:
                    load_path = f"./models/{m}/best_model.zip"
                    model = PPO.load(path=load_path, env=vec_env, device=device)  # don't need to pass policy_kwargs

                    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=eval_episodes)
                    print(f"Env:{env} Agent:{agent} Seed:{seed} Mean reward:{mean_reward:.2f} +/- {std_reward:.2f}")
                    df.loc[len(df.index)] = [env, agent, seed, mean_reward, std_reward]

        df.to_csv(path)
