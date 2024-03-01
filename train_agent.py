import sys

import wandb
from rl_zoo3.utils import linear_schedule
from skill_models import *
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
from feature_extractors import LinearConcatExtractor, CNNConcatExtractor, CombineExtractor
from wandb.integration.sb3 import WandbCallback
import os
import torch
import yaml

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use-skill", help="if True, use skill agent, otherwise standard agent",
                    type=bool, choices=[True, False])
parser.add_argument("--device", help="integer number of a device to use (0, 1, 2, 3), or cpu",
                    type=str, default="cpu", required=False, choices=["cpu", "0", "1", "2", "3"])
parser.add_argument("--env", help="Name of the environment to use i.e. Pong",
                    type=str)

parser.add_argument("--extractor", help="Which type of feature extractor to use", type=str,
                    default="lin_concat_ext", required=False,
                    choices=["lin_concat_ext", "cnn_concat_ext", "combine_ext"])

args = parser.parse_args()

skilled_agent = args.use_skill == True
tb_log_name = "PPO" if not skilled_agent else "SPPO"

device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
if not torch.cuda.is_available() and device != "cpu":
    print("CUDA not available, using CPU")
    device = "cpu"

env = args.env.lower()
env_name = args.env
with open(f'configs/{env}.yaml', 'r') as file:
    config = yaml.safe_load(file)["config"]

config["device"] = device
config["f_ext_kwargs"]["device"] = device
config["game"] = env_name

if args.extractor == "lin_concat_ext":
    config["f_ext_class"] = LinearConcatExtractor
    tb_log_name += "_lin"
    feature_dim = 16896
if args.extractor == "cnn_concat_ext":
    config["f_ext_class"] = CNNConcatExtractor
    tb_log_name += "_cnn"
    feature_dim = 8192
if args.extractor == "combine_ext":
    config["f_ext_class"] = CombineExtractor
    tb_log_name += "_comb"
    feature_dim = 8704

# run = wandb.init(
#     project = "sb3-skillcomp",
#     config = config,
#     sync_tensorboard = True,  # auto-upload sb3's tensorboard metrics
#     monitor_gym = True,  # auto-upload the videos of agents playing the game
#     name = f"{config['f_ext_name']}_{config['game']}",
#     tags=[config["game"].lower()]
#     # save_code = True,  # optional
# )

game_id = config["game"] + "NoFrameskip-v4"

logdir = "./tensorboard_logs"
gamelogs = f"{logdir}/{config['game']}"
if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(gamelogs):
    os.makedirs(gamelogs)

# vec_env = make_atari_env(game_id, n_envs=config["n_envs"], monitor_dir=f"monitor/{run.id}")
vec_env = make_atari_env(game_id, n_envs=config["n_envs"])
vec_env = VecFrameStack(vec_env, n_stack=config["n_stacks"])

skills = []
skills.append(get_state_rep_uns(config["game"], config["device"]))
skills.append(get_object_keypoints_encoder(config["game"], config["device"], load_only_model=True))
skills.append(get_object_keypoints_keynet(config["game"], config["device"], load_only_model=True))
skills.append(get_video_object_segmentation(config["game"], config["device"], load_only_model=True))

f_ext_kwargs = config["f_ext_kwargs"]
f_ext_kwargs["skills"] = skills
f_ext_kwargs["features_dim"] = feature_dim
if skilled_agent:
    policy_kwargs = dict(
        features_extractor_class=config["f_ext_class"],
        features_extractor_kwargs=f_ext_kwargs,
        net_arch={
            'pi': config["net_arch_pi"],
            'vf': config["net_arch_vf"]
        }
    )
else:
    policy_kwargs = None

model = PPO("CnnPolicy",
            vec_env,
            learning_rate=linear_schedule(config["learning_rate"]),

            n_steps=1, #n_steps=config["n_steps"],
            n_epochs=1, #n_epochs=config["n_epochs"],

            batch_size=config["batch_size"],
            clip_range=linear_schedule(config["clip_range"]),
            normalize_advantage=config["normalize"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            #tensorboard_log=gamelogs,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=config["device"],
            )

# print("net_arch:", model.policy.net_arch)
# print("share_feature_extractor:", model.policy.share_features_extractor)
# print("feature_extractor:", model.policy.features_extractor)
# print("num_skills:", len(model.policy.features_extractor.skills))
# for s in model.policy.features_extractor.skills:
#     print(s.name, "is training", s.skill_model.training)
# print("params:", sum(p.numel() for p in model.policy.parameters() if p.requires_grad))

#model.learn(config["n_timesteps"], tb_log_name=tb_log_name)
model.learn(1)

# model.learn(
#     config["n_timesteps"],
#     callback=WandbCallback(
#         model_save_path=f"models/{run.id}",
#         verbose=2,
#     )
# )
#
# run.finish()
