from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
import gym
from gym import spaces
from gym.envs.classic_control import CartPoleEnv
import numpy as np
import cv2


class CartPoleImageWrapper(CartPoleEnv):
    def __init__(self, *args, **kwargs):
        super(CartPoleImageWrapper, self).__init__(*args, **kwargs)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.render_mode = 'rgb_array'
    def _get_image_observation(self):
        # Render the CartPole environment
        cartpole_image = self.render()

        # Convert the image to grayscale
        grayscale_image = cv2.cvtColor(cartpole_image, cv2.COLOR_RGB2GRAY)

        # Resize the image to 84x84 pixels
        resized_image = cv2.resize(grayscale_image, (84, 84))

        return np.expand_dims(resized_image, axis=-1)

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        return self._get_image_observation()

    def step(self, action):
        observation, reward, terminated, done, info = super(CartPoleImageWrapper, self).step(action)
        return self._get_image_observation(), reward, done, info

# Create the custom CartPole environment
env = CartPoleImageWrapper()

# Wrap the environment in a vectorized form
vec_env = make_vec_env(lambda: env, n_envs=1)
vec_env = VecTransposeImage(vec_env)
vec_env = VecFrameStack(vec_env, n_stack=4)
obs = vec_env.reset()
print(f"Observation space: {obs.shape}")
# exit()
#vec_env.render("rgb_array")

action = vec_env.action_space.sample()
new_obs, rewards, dones, infos = vec_env.step([action])
print(f"Observation space: {new_obs.shape}")

action = vec_env.action_space.sample()
new_obs, rewards, dones, infos = vec_env.step([action])
print(f"Observation space: {new_obs.shape}")

action = vec_env.action_space.sample()
new_obs, rewards, dones, infos = vec_env.step([action])
print(f"Observation space: {new_obs.shape}")

action = vec_env.action_space.sample()
new_obs, rewards, dones, infos = vec_env.step([action])
print(f"Observation space: {new_obs.shape}")

# exit()
vec_env.close()
