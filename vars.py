from feature_extractors import LinearConcatExtractor, CNNConcatExtractor, CombineExtractor

dev = "cuda:1"

config = {
    "game": "Pong",
    "device": dev,

    "n_envs": 8,
    "n_stacks": 4,

    "n_steps": 128,
    "n_epochs": 4,
    "batch_size": 256,
    "n_timesteps": float(1e7),
    "learning_rate": 2.5e-4,
    "clip_range": 0.1,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
    "normalize": True,

    "net_arch_pi": [256],
    "net_arch_vf": [256],

    "f_ext_name": "lin_concat_ext",
    "f_ext_class": LinearConcatExtractor,
    "f_ext_ft_dim": 16896,
    "f_ext_kwargs": {
        "features_dim": 16896,
        "device": dev
    },

    # "f_ext_name" : "cnn_concat_ext",
    # "f_ext_class" : CNNConcatExtractor,
    #     "f_ext_kwargs": {
    #         "features_dim" : 8192,
    #         "num_conv_layers": 2,
    #         "device" : dev
    #     },

    # "f_ext_name" : "combine_ext",
    # "f_ext_class" : CombineExtractor,
    #     "f_ext_kwargs": {
    #         "features_dim" : 8704,
    #         "num_linear_skills": 1,
    #         "device" : dev
    #     },
}

# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml#L11C2-L11C28
# atari:
#   env_wrapper:
#     - stable_baselines3.common.atari_wrappers.AtariWrapper
#   frame_stack: 4
#   policy: 'CnnPolicy'
#   n_envs: 8
#   n_steps: 128
#   n_epochs: 4
#   batch_size: 256
#   n_timesteps: !!float 1e7
#   learning_rate: lin_2.5e-4
#   clip_range: lin_0.1
#   vf_coef: 0.5
#   ent_coef: 0.01
