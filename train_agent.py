# general imports
import torch as th
import yaml
import numpy as np
import random
import os
import pickle

# training imports
import wandb
from rl_zoo3.utils import linear_schedule
from skill_models import *
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

from feature_extractors import LinearConcatExtractor, FixedLinearConcatExtractor, \
    CNNConcatExtractor, CombineExtractor, \
    DotProductAttentionExtractor, WeightSharingAttentionExtractor, \
    ReservoirConcatExtractor

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, \
    StopTrainingOnNoModelImprovement
from wandb.integration.sb3 import WandbCallback
import tensorflow as tf

# utility imports
from utils.args import parse_args

from stable_baselines3 import DQN, PPO

# use this instead of classical PPO in case you want to use the custom PPO implementation with attention weights entropy loss, eventually change the coefficient
#from utils.custom_ppo import PPO

# ---------------------------------- MAIN ----------------------------------

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # ignore tensorflow warnings about CPU

args = parse_args()

seed = None
if args.seed is not None:
    seed = args.seed

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

skilled_agent = args.use_skill == "True"
alg = args.alg

tb_log_name = alg if not skilled_agent else "S" + alg
debug = args.debug == "True"

device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
if not torch.cuda.is_available() and device != "cpu":
    print("CUDA not available, using CPU")
    device = "cpu"

env = args.env.lower()
env_name = args.env
if alg == "DQN":
    with open("configs/atari_dqn.yaml", 'r') as file:
        config = yaml.safe_load(file)["config"]
else:
    with open(f'configs/{env}.yaml', 'r') as file:
        config = yaml.safe_load(file)["config"]

if "_" in env_name:
    env_name = env_name.replace("_", "")

config["f_ext_kwargs"]["device"] = device  #do not comment this, it is the parameter passed to the feature extractor
config["game"] = env_name
config["net_arch_pi"] = args.pi
config["net_arch_vf"] = args.vf

expert = args.use_expert == "True"

version = "3-5 entropy"     # change for future experiments

tags = [f'game:{config["game"]}', f'version:{version}', f'seed:{seed}', f'alg:{alg}']

string = "pi:"
for el in config["net_arch_pi"]:
    string += str(el) + "-"
tags.append(string)

string = "vf:"
for el in config["net_arch_vf"]:
    string += str(el) + "-"
tags.append(string)

game_id = env_name + "NoFrameskip-v4"

logdir = "./tensorboard_logs"
gamelogs = f"{logdir}/{env_name}"
if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(gamelogs):
    os.makedirs(gamelogs)

vec_env = make_atari_env(game_id, n_envs=config["n_envs"], seed=seed)
vec_env = VecFrameStack(vec_env, n_stack=config["n_stacks"])
vec_env = VecTransposeImage(vec_env)

skills = []
skills.append(get_state_rep_uns(env_name, device, expert=expert))
skills.append(get_object_keypoints_encoder(env_name, device, load_only_model=True, expert=expert))
skills.append(get_object_keypoints_keynet(env_name, device, load_only_model=True, expert=expert))
skills.append(get_video_object_segmentation(env_name, device, load_only_model=True, expert=expert))

# skills.append(get_autoencoder(env_name, device, expert=expert))   # autoencoder skill used also in attention mechanism extractors as context
# skills.append(get_image_completion(env_name, device, expert=expert))
# skills.append(get_frame_prediction(env_name, device, expert=expert))

f_ext_kwargs = config["f_ext_kwargs"]
sample_obs = vec_env.observation_space.sample()
sample_obs = torch.tensor(sample_obs).to(device)
sample_obs = sample_obs.unsqueeze(0)
# print("sample obs shape", sample_obs.shape)

features_dim = 256
if skilled_agent:
    tags.append(f'ext:{args.extractor}')
    if args.extractor == "lin_concat_ext":
        config["f_ext_name"] = "lin_concat_ext"
        config["f_ext_class"] = LinearConcatExtractor
        tb_log_name += "_lin"
        ext = LinearConcatExtractor(observation_space=vec_env.observation_space, skills=skills, device=device)
        features_dim = ext.get_dimension(sample_obs)

    if args.extractor == "fixed_lin_concat_ext":
        config["f_ext_name"] = "fixed_lin_concat_ext"
        config["f_ext_class"] = FixedLinearConcatExtractor
        f_ext_kwargs["fixed_dim"] = args.fd
        tags.append(f"fixed_dim:{f_ext_kwargs['fixed_dim']}")
        tb_log_name += "_fixedlin"
        ext = FixedLinearConcatExtractor(observation_space=vec_env.observation_space, skills=skills, device=device,
                                         fixed_dim=f_ext_kwargs["fixed_dim"])
        features_dim = ext.get_dimension(sample_obs)

    if args.extractor == "cnn_concat_ext":
        config["f_ext_name"] = "cnn_concat_ext"
        config["f_ext_class"] = CNNConcatExtractor
        f_ext_kwargs["num_conv_layers"] = args.cv
        tb_log_name += "_cnn"
        ext = CNNConcatExtractor(vec_env.observation_space, skills=skills, device=device)
        features_dim = ext.get_dimension(sample_obs)
        tags.append(f"cnn_layers:{args.cv}")

    if args.extractor == "combine_ext":
        config["f_ext_name"] = "combine_ext"
        config["f_ext_class"] = CombineExtractor
        f_ext_kwargs["num_linear_skills"] = 1
        tb_log_name += "_comb"
        ext = CombineExtractor(vec_env.observation_space, skills=skills, device=device,
                               num_linear_skills=f_ext_kwargs["num_linear_skills"])
        features_dim = ext.get_dimension(sample_obs)

    if args.extractor == "dotproduct_attention_ext":
        config["f_ext_name"] = "dotproduct_attention_ext"
        config["f_ext_class"] = DotProductAttentionExtractor
        features_dim = args.fd
        tb_log_name += "_dpae"
        f_ext_kwargs["game"] = env_name
        f_ext_kwargs["expert"] = expert
        tags.append(f"fixed_dim:{features_dim}")

    if args.extractor == "wsharing_attention_ext":
        config["f_ext_name"] = "wsharing_attention_ext"
        config["f_ext_class"] = WeightSharingAttentionExtractor
        features_dim = args.fd
        tb_log_name += "_wsae"
        f_ext_kwargs["game"] = env_name
        f_ext_kwargs["expert"] = expert
        tags.append(f"fixed_dim:{features_dim}")

    if args.extractor == "reservoir_concat_ext":
        config["f_ext_name"] = "reservoir_concat_ext"
        config["f_ext_class"] = ReservoirConcatExtractor
        tb_log_name += "_reservoir"
        max_batch_size = max(config["net_arch_pi"][0], config["net_arch_vf"][0])
        f_ext_kwargs["max_batch_size"] = max_batch_size

        # dato che concateno le skill come nel linear, uso LinearConcatExt per prendere la dimensione
        ext = LinearConcatExtractor(vec_env.observation_space, skills=skills, device=device)
        input_features_dim = ext.get_dimension(sample_obs)
        reservoir_output_dim = args.ro  # output dimension of the reservoir WARNING: if it is too big memory error
        features_dim = reservoir_output_dim
        f_ext_kwargs["input_features_dim"] = input_features_dim
        tags.append(f"res_size:{args.ro}")

f_ext_kwargs["skills"] = skills
f_ext_kwargs["features_dim"] = features_dim

if skilled_agent:
    if alg == "PPO":
        policy_kwargs = dict(
            features_extractor_class=config["f_ext_class"],
            features_extractor_kwargs=f_ext_kwargs,
            net_arch={
                'pi': config["net_arch_pi"],
                'vf': config["net_arch_vf"]
            },
            #activation_fn=th.nn.ReLU,  # use ReLU in case of multiple layers for the policy learning network

        )
    if alg == "DQN":
        policy_kwargs = dict(
            features_extractor_class=config["f_ext_class"],
            features_extractor_kwargs=f_ext_kwargs,
        )
else:
    policy_kwargs = None

if debug:
    if alg == "PPO":
        model = PPO("CnnPolicy",
                    vec_env,
                    learning_rate=linear_schedule(config["learning_rate"]),
                    n_steps=128,
                    n_epochs=4,
                    batch_size=config["batch_size"],
                    clip_range=linear_schedule(config["clip_range"]),
                    normalize_advantage=config["normalize"],
                    ent_coef=config["ent_coef"],
                    vf_coef=config["vf_coef"],
                    policy_kwargs=policy_kwargs,
                    verbose=0,
                    device=device,
                    )

    if alg == "DQN":
        model = DQN("CnnPolicy",
                    vec_env,
                    buffer_size=config["buffer_size"],
                    learning_rate=config["learning_rate"],
                    batch_size=config["batch_size"],
                    learning_starts=config["learning_starts"],
                    target_update_interval=config["target_update_interval"],
                    train_freq=config["train_freq"],
                    gradient_steps=config["gradient_steps"],
                    exploration_fraction=config["exploration_fraction"],
                    exploration_final_eps=config["exploration_final_eps"],
                    optimize_memory_usage=config["optimize_memory_usage"],
                    policy_kwargs=policy_kwargs,
                    verbose=0,
                    device=device,
                    )

    model.learn(1000)



else:
    run = wandb.init(
        project="sb3-skillcomp",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        name=f"{config['f_ext_name']}__{config['game']}",
        group=f"{config['game']}",
        tags=tags
        # save_code = True,  # optional
    )

    vec_env = make_atari_env(game_id, n_envs=config["n_envs"], monitor_dir=f"monitor/{run.id}")
    vec_env = VecFrameStack(vec_env, n_stack=config["n_stacks"])
    vec_env = VecTransposeImage(vec_env)

    vec_eval_env = make_atari_env(game_id, n_envs=config["n_envs"])
    vec_eval_env = VecFrameStack(vec_eval_env, n_stack=config["n_stacks"])
    vec_eval_env = VecTransposeImage(vec_eval_env)


    if alg == "PPO":
        model = PPO("CnnPolicy",
                    vec_env,
                    learning_rate=linear_schedule(config["learning_rate"]),
                    n_steps=128,
                    n_epochs=4,
                    batch_size=config["batch_size"],
                    clip_range=linear_schedule(config["clip_range"]),
                    normalize_advantage=config["normalize"],
                    ent_coef=config["ent_coef"],
                    vf_coef=config["vf_coef"],
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=gamelogs,
                    verbose=0,
                    device=device,
                    )

    if alg == "DQN":
        model = DQN("CnnPolicy",
                    vec_env,
                    buffer_size=config["buffer_size"],
                    learning_rate=config["learning_rate"],
                    batch_size=config["batch_size"],
                    learning_starts=config["learning_starts"],
                    target_update_interval=config["target_update_interval"],
                    train_freq=config["train_freq"],
                    gradient_steps=config["gradient_steps"],
                    exploration_fraction=config["exploration_fraction"],
                    exploration_final_eps=config["exploration_final_eps"],
                    optimize_memory_usage=config["optimize_memory_usage"],
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=gamelogs,
                    verbose=0,
                    device=device,
                    )

    # model.learn(config["n_timesteps"], tb_log_name=tb_log_name)

    #stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=5, verbose=0)
    # if env_name == "Pong":
    #     if not skilled_agent:
    #         eval_callback = EvalCallback(
    #             vec_eval_env,
    #             n_eval_episodes=10,
    #             best_model_save_path=f"models/{run.id}",
    #             log_path=gamelogs,
    #             eval_freq=5000 * config["n_envs"],
    #             verbose=0
    #         )
    #     else:
    #         callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=21, verbose=0)
    #         eval_callback = EvalCallback(
    #             vec_eval_env,
    #             n_eval_episodes=10,
    #             best_model_save_path=f"models/{run.id}",
    #             log_path=gamelogs,
    #             eval_freq=5000 * config["n_envs"],
    #             callback_on_new_best=callback_on_best,
    #             callback_after_eval=stop_train_callback,
    #             verbose=0
    #
    #         )
    # else:
    #     if not skilled_agent:
    #         eval_callback = EvalCallback(
    #             vec_eval_env,
    #             n_eval_episodes=10,
    #             best_model_save_path=f"models/{run.id}",
    #             log_path=gamelogs,
    #             eval_freq=5000 * config["n_envs"],
    #             verbose=0
    #
    #         )
    #     else:
    #         eval_callback = EvalCallback(
    #             vec_eval_env,
    #             n_eval_episodes=10,
    #             best_model_save_path=f"models/{run.id}",
    #             log_path=gamelogs,
    #             eval_freq=5000 * config["n_envs"],
    #             callback_after_eval=stop_train_callback,
    #             verbose=0
    #         )

    eval_callback = EvalCallback(
        vec_eval_env,
        n_eval_episodes=100,
        best_model_save_path=f"models/{run.id}",
        log_path=gamelogs,
        eval_freq=5000 * config["n_envs"],
        verbose=0
    )

    callbacks = [
        WandbCallback(
            verbose=0
        ),
        eval_callback
    ]

    model.learn(
        config["n_timesteps"],
        callback=callbacks
    )
    run.finish()


# in case you want to save the attention training weights in order to visualize them
# if hasattr(model.policy.features_extractor, "training_weights"):
#     training_weights = model.policy.features_extractor.training_weights
#
#     with open(f'{env}_attention_weights.pkl', 'wb') as f:
#         pickle.dump(training_weights, f)


# in case you want to print some information about the model like the number of parameters
# print("net_arch:", model.policy.net_arch)
# print("share_feature_extractor:", model.policy.share_features_extractor)
# print("feature_extractor:", model.policy.features_extractor)
#
# if skilled_agent:
#     print("num_skills:", len(model.policy.features_extractor.skills))
#     for s in model.policy.features_extractor.skills:
#         print(s.name, "is training", s.skill_model.training)
#
# print("params:", sum(p.numel() for p in model.policy.parameters() if p.requires_grad))
