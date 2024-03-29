import sys
import wandb
from rl_zoo3.utils import linear_schedule
from skill_models import *
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3 import PPO
from feature_extractors import LinearConcatExtractor, CNNConcatExtractor, CombineExtractor, SelfAttentionExtractor, \
    ReservoirConcatExtractor
from wandb.integration.sb3 import WandbCallback
import os
import torch
import yaml

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use-skill", help="if True, use skill agent, otherwise standard agent",
                    type=str, choices=["True", "False"], required=True)
parser.add_argument("--device", help="integer number of a device to use (0, 1, 2, 3), or cpu",
                    type=str, default="cpu", required=False, choices=["cpu", "0", "1", "2", "3"])
parser.add_argument("--env", help="Name of the environment to use i.e. Pong",
                    type=str, required=True, choices=["Pong", "Breakout"])
parser.add_argument("--extractor", help="Which type of feature extractor to use", type=str,
                    default="lin_concat_ext", required=False,
                    choices=["lin_concat_ext", "cnn_concat_ext", "combine_ext",
                             "self_attention_ext", "reservoir_concat_ext"])

parser.add_argument("--debug", type=str, default="False", choices=["True", "False"])

args = parser.parse_args()

skilled_agent = args.use_skill == "True"

tb_log_name = "PPO" if not skilled_agent else "SPPO"
debug = args.debug == "True"

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
config["game"] = env_name+"_ae"

game_id = env_name + "NoFrameskip-v4"

logdir = "./tensorboard_logs"
gamelogs = f"{logdir}/{env_name}"
if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(gamelogs):
    os.makedirs(gamelogs)

vec_env = make_atari_env(game_id, n_envs=config["n_envs"])
vec_env = VecFrameStack(vec_env, n_stack=config["n_stacks"])
vec_env = VecTransposeImage(vec_env)

skills = []
skills.append(get_state_rep_uns(env_name, config["device"]))
skills.append(get_object_keypoints_encoder(env_name, config["device"], load_only_model=True))
skills.append(get_object_keypoints_keynet(env_name, config["device"], load_only_model=True))
skills.append(get_video_object_segmentation(env_name, config["device"], load_only_model=True))
skills.append(get_autoencoder(env_name, config["device"]))
#skills.append(get_image_completion(env_name, config["device"]))

f_ext_kwargs = config["f_ext_kwargs"]
sample_obs = vec_env.observation_space.sample()
sample_obs = torch.tensor(sample_obs).to(device)
sample_obs = sample_obs.unsqueeze(0)
# print("sample obs shape", sample_obs.shape)

features_dim = 256
if skilled_agent:
    if args.extractor == "lin_concat_ext":
        config["f_ext_name"] = "lin_concat_ext"
        config["f_ext_class"] = LinearConcatExtractor
        tb_log_name += "_lin"
        ext = LinearConcatExtractor(vec_env.observation_space, skills=skills, device=device)
        features_dim = ext.get_dimension(sample_obs)

    if args.extractor == "cnn_concat_ext":
        config["f_ext_name"] = "cnn_concat_ext"
        config["f_ext_class"] = CNNConcatExtractor
        tb_log_name += "_cnn"
        ext = CNNConcatExtractor(vec_env.observation_space, skills=skills, device=device)
        features_dim = ext.get_dimension(sample_obs)

    if args.extractor == "combine_ext":
        config["f_ext_name"] = "combine_ext"
        config["f_ext_class"] = CombineExtractor
        tb_log_name += "_comb"
        ext = CombineExtractor(vec_env.observation_space, skills=skills, device=device, num_linear_skills=0)
        features_dim = ext.get_dimension(sample_obs)

    if args.extractor == "self_attention_ext":
        config["f_ext_name"] = "self_attention_ext"
        config["f_ext_class"] = SelfAttentionExtractor
        tb_log_name += "_sae"
        f_ext_kwargs["n_features"] = 256
        f_ext_kwargs["n_heads"] = 4
        features_dim = len(skills) * f_ext_kwargs["n_features"]

    if args.extractor == "reservoir_concat_ext":
        config["f_ext_name"] = "reservoir_concat_ext"
        config["f_ext_class"] = ReservoirConcatExtractor
        tb_log_name += "_reservoir"
        max_batch_size = config["net_arch_pi"][0] #config["net_arch_vf"] #controlla PPO, vedi se prima utilizza pi e dopo vf, potrebbe dare errore
        f_ext_kwargs["max_batch_size"] = max_batch_size

        # dato che concateno le skill come nel linear, uso LinearConcatExt per prendere la dimensione
        ext = LinearConcatExtractor(vec_env.observation_space, skills=skills, device=device)
        input_features_dim = ext.get_dimension(sample_obs)
        features_dim = len(skills) * 256  # output dimension of the reservoir WARNING: if it is too big memory error
        f_ext_kwargs["input_features_dim"] = input_features_dim

f_ext_kwargs["skills"] = skills
f_ext_kwargs["features_dim"] = features_dim

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

if debug:
    model = PPO("CnnPolicy",
                vec_env,
                learning_rate=linear_schedule(config["learning_rate"]),
                n_steps=1,
                n_epochs=1,
                batch_size=config["batch_size"],
                clip_range=linear_schedule(config["clip_range"]),
                normalize_advantage=config["normalize"],
                ent_coef=config["ent_coef"],
                vf_coef=config["vf_coef"],
                policy_kwargs=policy_kwargs,
                verbose=0,
                device=config["device"],
                )
    model.learn(1000)
else:
    run = wandb.init(
        project="sb3-skillcomp",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        name=f"{config['f_ext_name']}_{config['game']}",
        tags=[config["game"].lower()]
        # save_code = True,  # optional
    )

    vec_env = make_atari_env(game_id, n_envs=config["n_envs"], monitor_dir=f"monitor/{run.id}")
    vec_env = VecFrameStack(vec_env, n_stack=config["n_stacks"])
    vec_env = VecTransposeImage(vec_env)

    model = PPO("CnnPolicy",
                vec_env,
                learning_rate=linear_schedule(config["learning_rate"]),
                n_steps=config["n_steps"],
                n_epochs=config["n_epochs"],
                batch_size=config["batch_size"],
                clip_range=linear_schedule(config["clip_range"]),
                normalize_advantage=config["normalize"],
                ent_coef=config["ent_coef"],
                vf_coef=config["vf_coef"],
                tensorboard_log=gamelogs,
                policy_kwargs=policy_kwargs,
                verbose=1,
                device=config["device"],
                )

    #model.learn(config["n_timesteps"], tb_log_name=tb_log_name)
    model.learn(
        config["n_timesteps"],
        callback=WandbCallback(
            model_save_path=f"models/{run.id}",
            verbose=2,
        )
    )
    run.finish()

# print("net_arch:", model.policy.net_arch)
# print("share_feature_extractor:", model.policy.share_features_extractor)
# print("feature_extractor:", model.policy.features_extractor)
# print("num_skills:", len(model.policy.features_extractor.skills))
# for s in model.policy.features_extractor.skills:
#     print(s.name, "is training", s.skill_model.training)
# print("params:", sum(p.numel() for p in model.policy.parameters() if p.requires_grad))