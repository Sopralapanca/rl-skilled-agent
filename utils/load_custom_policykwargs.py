from skill_models import *
from feature_extractors import LinearConcatExtractor, FixedLinearConcatExtractor, \
    CNNConcatExtractor, CombineExtractor, \
    DotProductAttentionExtractor, WeightSharingAttentionExtractor, \
    ReservoirConcatExtractor
import yaml
from typing import List
import torch as th
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage


def load_policy_kwargs(expert: bool, device: str, env: str,
                       net_arch: List[int], agent: str,
                       features_dim: int, num_conv_layers: int) -> dict:
    """
    Load the policy kwargs for the given environment and change the device to the given device.
    :param expert: bool: If True, loads saved skills trained on expert data
    :param device: str: Device to use
    :param env: str: Environment name like Pong, Breakout, Ms_Pacman, etc.
    :param net_arch: List[int]: List of integers representing the number of units in each layer of the policy network
    :param agent: str: Name of the agent to use like wsharing_attention_ext, reservoir_concat_ext, etc.
    :param features_dim int: Dimension of the features extracted by the feature extractor
    :param num_conv_layers: int: Number of convolutional layers to use in cnn_concat_ext

    :return: dict: Custom objects to be used in the model
    """

    env_name = env.split("-")[0]
    if "_" in env_name:
        vec_env_name = env_name.replace("_", "")
    else:
        vec_env_name = env_name

    vec_env = make_atari_env(f"{vec_env_name}NoFrameskip-v4", n_envs=1)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env = VecTransposeImage(vec_env)

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

    config["net_arch_pi"] = net_arch
    config["net_arch_vf"] = net_arch

    config["f_ext_name"] = agent
    if agent == "wsharing_attention_ext":
        config["f_ext_class"] = WeightSharingAttentionExtractor
        config["game"] = vec_env_name

    elif agent == "reservoir_concat_ext":
        config["f_ext_class"] = ReservoirConcatExtractor
        ext = LinearConcatExtractor(vec_env.observation_space, skills=skills, device=device)
        input_features_dim = ext.get_dimension(sample_obs)

    elif agent == "cnn_concat_ext":
        ext = CNNConcatExtractor(vec_env.observation_space, skills=skills, device=device)
        features_dim = ext.get_dimension(sample_obs)
        config["f_ext_class"] = CNNConcatExtractor

    elif agent == "fixed_lin_concat_ext":
        config["f_ext_class"] = FixedLinearConcatExtractor
        ext = FixedLinearConcatExtractor(observation_space=vec_env.observation_space, skills=skills,
                                         device=device, fixed_dim=features_dim)
        features_dim = ext.get_dimension(sample_obs)

    f_ext_kwargs = config["f_ext_kwargs"]

    if agent == "wsharing_attention_ext":
        f_ext_kwargs["game"] = vec_env_name
        f_ext_kwargs["expert"] = expert

    elif agent == "reservoir_concat_ext":
        f_ext_kwargs["input_features_dim"] = input_features_dim

    elif agent == "cnn_concat_ext":
        if env_name == "Breakout":
            f_ext_kwargs["num_conv_layers"] = num_conv_layers
        else:
            f_ext_kwargs["num_conv_layers"] = num_conv_layers

    elif agent == "fixed_lin_concat_ext":
        f_ext_kwargs["fixed_dim"] = features_dim

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

    return custom_objects
