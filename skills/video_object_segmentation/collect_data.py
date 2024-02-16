import numpy as np
import os
import sys
from avalanche_rl.training.strategies.env_wrappers import ClipRewardWrapper, \
    FireResetWrapper, FrameStackingWrapper
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from avalanche_rl.benchmarks.rl_benchmark_generators import make_env
from PIL import Image

def make_dataset(env_name):
    # Hyper-parameters
    total_frames_to_generate = 100000
    env_id = env_name
    save_path = f'data/{env_name}'
    seed = 123

    # Track how many frames we have created.
    total_frames_generated = 0
    episode_index = 0

    wrappers = [
        AtariPreprocessing,
        FireResetWrapper,
        ClipRewardWrapper
    ]

    # Create and set-up the environment.
    env = make_env(env_id, wrappers=wrappers)
    print(env)
    env.seed(seed)
    # set_global_seeds(seed)

    # Generate frames.
    while total_frames_generated < total_frames_to_generate:
        print("Starting episode {} - Total frames generated: {}".format(episode_index, total_frames_generated))

        obs = env.reset()
        frame_index = 0
        done = False

        while not done and total_frames_generated < total_frames_to_generate:
            # Take a random action.
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            # Create a directory to save frames to for this episode.
            episode_save_path = os.path.join(save_path, str(episode_index))
            if not os.path.exists(episode_save_path):
                os.makedirs(episode_save_path)

            # Save the frame
            img = Image.fromarray(np.squeeze(obs), mode='L')
            img.save(os.path.join(episode_save_path, '{}_{}_{}.png'.format(frame_index, action, reward)))
            frame_index += 1
            total_frames_generated += 1

        # Start a new episode.
        episode_index += 1

def make_dataset_multigames():
    envs = [
    #train
    'BreakoutNoFrameskip-v4',
    'FishingDerbyNoFrameskip-v4',
    'FreewayNoFrameskip-v4',
    'GravitarNoFrameskip-v4',
    'KangarooNoFrameskip-v4',
    'MontezumaRevengeNoFrameskip-v4',
    'MsPacmanNoFrameskip-v4',
    'RobotankNoFrameskip-v4',
    'SpaceInvadersNoFrameskip-v4',
    'VideoPinballNoFrameskip-v4',
    #eval
    'PongNoFrameskip-v4'
    ]

    # Hyper-parameters
    total_frames_to_generate = 10000
    save_path = 'data'
    seed = 0

    for i, e in enumerate(envs):
        env = WarpFrame(make_atari(e, max_episode_steps=500), grayscale=True)
        env.seed(seed)
        frame_generated = 0
        ep_idx = 0

        while frame_generated < total_frames_to_generate:
            print("Env {} - Starting episode {} - Total frames generated: {}".format(i, ep_idx, frame_generated))
            obs = env.reset()
            frame_idx = 0
            done = False

            while not done and frame_generated < total_frames_to_generate:
                action = env.action_space.sample()
                obs, _, done, _ = env.step(action)

                episode_save_path = os.path.join(save_path, f'{i}', f'{ep_idx}')
                if not os.path.exists(episode_save_path):
                    os.makedirs(episode_save_path)
                
                img = Image.fromarray(np.squeeze(obs), mode='L')
                img.save(os.path.join(episode_save_path, f'{frame_idx}.png'))
                frame_idx += 1
                frame_generated += 1
            ep_idx += 1

if __name__ == '__main__':
    make_dataset(sys.argv[1] + 'NoFrameskip-v4')