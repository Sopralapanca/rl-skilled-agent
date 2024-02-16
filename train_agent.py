import wandb
from rl_zoo3.utils import linear_schedule
from skill_models import *
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
from vars import config
from wandb.integration.sb3 import WandbCallback

run = wandb.init(
    project = "sb3-skillcomp",
    config = config,
    sync_tensorboard = True,  # auto-upload sb3's tensorboard metrics
    monitor_gym = True,  # auto-upload the videos of agents playing the game
    name = f"{config['f_ext_name']}_{config['game']}",
    tags=[config["game"].lower()]
    # save_code = True,  # optional
)

game_id = config["game"] + "NoFrameskip-v4"
vec_env = make_atari_env(game_id, n_envs=config["n_envs"], monitor_dir=f"monitor/{run.id}")
vec_env = VecFrameStack(vec_env, n_stack=config["n_stacks"])

skills = []
skills.append(get_state_rep_uns(config["game"], config["device"]))
skills.append(get_object_keypoints_encoder(config["game"], config["device"], load_only_model=True))
skills.append(get_object_keypoints_keynet(config["game"], config["device"], load_only_model=True))
skills.append(get_video_object_segmentation(config["game"], config["device"], load_only_model=True))

f_ext_kwargs = config["f_ext_kwargs"]
f_ext_kwargs["skills"] = skills

policy_kwargs = dict(
    features_extractor_class=config["f_ext_class"],
    features_extractor_kwargs=f_ext_kwargs,
    net_arch = {
        'pi' : config["net_arch_pi"],
        'vf' : config["net_arch_vf"]
    }
)

model = PPO("CnnPolicy",
            vec_env,
            learning_rate=linear_schedule(config["learning_rate"]),
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            clip_range=linear_schedule(config["clip_range"]),
            normalize_advantage=config["normalize"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            tensorboard_log=f"./logs/ppo_{config['f_ext_name']}_{config['game']}",
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=config["device"],
        )

# print("net_arch:", model.policy.net_arch)
# print("share_feature_extractor:", model.policy.share_features_extractor)
# print("feature_extractor:", model.policy.features_extractor)
# print("num_skills:", len(model.policy.features_extractor.skills))
# for s in model.policy.features_extractor.skills:
#     print(s.name, "is training", s.skill_model.training)
# print("params:", sum(p.numel() for p in model.policy.parameters() if p.requires_grad))

# model.learn(config["n_timesteps"])

model.learn(
    config["n_timesteps"],
    callback=WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
    )
)

run.finish()