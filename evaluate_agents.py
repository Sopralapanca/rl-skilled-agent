# general imports
import torch
import yaml
import numpy as np
import random
import os

# testing imports
import wandb
from rl_zoo3.utils import linear_schedule
from skill_models import *
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3 import PPO
from feature_extractors import LinearConcatExtractor, FixedLinearConcatExtractor, \
    CNNConcatExtractor, CombineExtractor, \
    SelfAttentionExtractor, DotProductAttentionExtractor, WeightSharingAttentionExtractor, SelfAttentionExtractor2, \
    ReservoirConcatExtractor

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, \
    StopTrainingOnNoModelImprovement
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback
import tensorflow as tf
from stable_baselines3.common.vec_env import VecVideoRecorder

# utility imports
from utils.args import parse_args

# ---------------------------------- MAIN ----------------------------------

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # ignore tensorflow warnings about CPU
n_seeds = 1
seeds = [np.random.randint(0, 100000) for i in range(n_seeds)]

device = f"cuda:{0}"
envs = ["Asteroids", "Pong", "MsPacman"]

results_dict = {"Pong": {"PPO": [],
                         "wsharing_attention_ext": [],
                         "reservoir_concat_ext": [],
                         "cnn_concat_ext": [],
                         },
                "Asteroids": {"PPO": [],
                              "reservoir_concat_ext": [],
                              },
                "MsPacman": {"PPO": [],
                             "wsharing_attention_ext": [],
                             "reservoir_concat_ext": [],
                             "cnn_concat_ext": [],
                             }}

eval_episodes = 2
for env in envs:
    with open(f'configs/asteroids.yaml', 'r') as file:
        config = yaml.safe_load(file)["config"]

    #config["device"] = device
    config["f_ext_kwargs"]["device"] = device
    config["game"] = env
    config["net_arch_pi"] = [256]
    config["net_arch_vf"] = [256]

    skills = []
    skills.append(get_state_rep_uns(env, device))
    skills.append(get_object_keypoints_encoder(env, device, load_only_model=True))
    skills.append(get_object_keypoints_keynet(env, device, load_only_model=True))
    skills.append(get_video_object_segmentation(env, device, load_only_model=True))

    f_ext_kwargs = config["f_ext_kwargs"]

    if env == "Asteroids":
        extractors = ["reservoir_concat_ext"]
    if env == "Pong" or env == "MsPacman":
        extractors = ["wsharing_attention_ext", "reservoir_concat_ext", "cnn_concat_ext"]

    load_path = ""
    atari_env = env + "NoFrameskip-v4"
    for e in extractors:
        for seed in seeds:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            vec_env = make_atari_env(atari_env, n_envs=config["n_envs"], seed=seed)
            vec_env = VecFrameStack(vec_env, n_stack=config["n_stacks"])
            vec_env = VecTransposeImage(vec_env)
            # vec_env = VecVideoRecorder(vec_env,
            #                            f"./",
            #                            record_video_trigger=lambda x: x % 2000 == 0,
            #                            video_length=200,
            #                            )

            # test standard PPO
            if env == "Asteroids":
                load_path = f"./models/nx6ffjli/best_model.zip"
            elif env == "Pong":
                load_path = f"./models/sco059rf/best_model.zip"
            elif env == "MsPacman":
                load_path = f"./models/pvascv4i/best_model.zip"

            model = PPO.load(path=load_path, env=vec_env, device=device)  # don't need to pass policy_kwargs
            mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=eval_episodes)
            print(f"Env: {env} Agent: PPO Seed: {seed} Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
            results_dict[env]["PPO"].append([mean_reward, std_reward])


            sample_obs = vec_env.observation_space.sample()
            sample_obs = torch.tensor(sample_obs).to(device)
            sample_obs = sample_obs.unsqueeze(0)

            if e == "cnn_concat_ext":
                if env == "Pong":
                    load_path = f"./models/0nht2mrd/best_model.zip"
                elif env == "MsPacman":
                    load_path = f"./models/0vm9cdpz/best_model.zip"

                config["f_ext_name"] = "cnn_concat_ext"
                config["f_ext_class"] = CNNConcatExtractor
                f_ext_kwargs["num_conv_layers"] = 2
                ext = CNNConcatExtractor(vec_env.observation_space, skills=skills, device=device)
                features_dim = ext.get_dimension(sample_obs)

            if e == "wsharing_attention_ext":
                if env == "Pong":
                    features_dim = 1024
                    load_path = f"./models/630m4glh/best_model.zip"
                if env == "MsPacman":
                    features_dim = 256
                    load_path = f"./models/xbmyz15p/best_model.zip"

                config["f_ext_name"] = "wsharing_attention_ext"
                config["f_ext_class"] = WeightSharingAttentionExtractor
                f_ext_kwargs["game"] = env

            if e == "reservoir_concat_ext":
                if env == "Asteroids":
                    load_path = f"./models/x34mrjhm/best_model.zip"
                elif env == "Pong":
                    load_path = f"./models/ijaap571/best_model.zip"
                elif env == "MsPacman":
                    load_path = f"./models/88rmd7an/best_model.zip"

                config["f_ext_name"] = "reservoir_concat_ext"
                config["f_ext_class"] = ReservoirConcatExtractor
                max_batch_size = max(config["net_arch_pi"][0], config["net_arch_vf"][0])
                f_ext_kwargs["max_batch_size"] = max_batch_size

                # dato che concateno le skill come nel linear, uso LinearConcatExt per prendere la dimensione
                ext = LinearConcatExtractor(vec_env.observation_space, skills=skills, device=device)
                input_features_dim = ext.get_dimension(sample_obs)
                reservoir_output_dim = 1024  # output dimension of the reservoir WARNING: if it is too big memory error
                features_dim = reservoir_output_dim
                f_ext_kwargs["input_features_dim"] = input_features_dim

            f_ext_kwargs["skills"] = skills
            f_ext_kwargs["features_dim"] = features_dim

            model = PPO.load(path=load_path, env=vec_env, device=device)  # don't need to pass policy_kwargs

            # Evaluate the agent
            mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=eval_episodes)

            print(f"Env: {env} Agent: {e} Seed: {seed} Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
            results_dict[env][e].append([mean_reward, std_reward])
        print()


for env in envs:
    for key in results_dict[env].keys():
        mean_reward = sum(sublist[0] for sublist in l) / n_seeds
        mean_std = sum(sublist[2] for sublist in l) / n_seeds
        print(f"Env: {env} Agent: {key} Mean reward: {mean_reward:.2f} +/- {mean_std:.2f}")