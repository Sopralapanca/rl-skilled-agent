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
    DotProductAttentionExtractor, WeightSharingAttentionExtractor, \
    ReservoirConcatExtractor

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, \
    StopTrainingOnNoModelImprovement
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback
import tensorflow as tf
from stable_baselines3.common.vec_env import VecVideoRecorder

# utility imports
from utils.args import parse_args
import pandas as pd

# ---------------------------------- MAIN ----------------------------------

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # ignore tensorflow warnings about CPU
n_seeds = 10
seeds = [np.random.randint(0, 100000) for i in range(n_seeds)]

device = f"cuda:{3}"
eval_episodes = 1000

df = pd.DataFrame(columns=["env", "agent", "seed", "mean_reward", "std_reward"])
f"./models/630m4glh/best_model.zip"
d = {"Pong": {"PPO": "sco059rf",
              "wsharing_attention_ext": "630m4glh",
              "reservoir_concat_ext": "ijaap571",
              "cnn_concat_ext": "0nht2mrd",
              },
     "MsPacman": {"PPO": "pvascv4i",
                  "wsharing_attention_ext": "xbmyz15p",
                  "reservoir_concat_ext": "88rmd7an",
                  "cnn_concat_ext": "0vm9cdpz",
                  },
     }
for seed in seeds:
    for env in d.keys():
        c = env.lower()
        if env == "MsPacman":
            c = "ms_pacman"

        with open(f'configs/{c}.yaml', 'r') as file:
            config = yaml.safe_load(file)["config"]

        skills = []
        skills.append(get_state_rep_uns(env, device))
        skills.append(get_object_keypoints_encoder(env, device, load_only_model=True))
        skills.append(get_object_keypoints_keynet(env, device, load_only_model=True))
        skills.append(get_video_object_segmentation(env, device, load_only_model=True))

        atari_env = env + "NoFrameskip-v4"

        for e in d[env].keys():
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
            load_path = f"./models/{d[env][e]}/best_model.zip"
            print(f"Env:{env} Agent:{e} model path: {load_path}")

            model = PPO.load(path=load_path, env=vec_env, device=device)  # don't need to pass policy_kwargs
            mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=eval_episodes)
            print(f"Env: {env} Agent:{e} Seed: {seed} Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

            df.loc[len(df.index)] = [env, e, seed, mean_reward, std_reward]


df.to_csv("eval_results.csv")