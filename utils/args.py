import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-skill", help="if True, use skill agent, otherwise standard agent",
                        type=str, choices=["True", "False"], required=True)
    parser.add_argument("--device", help="Integer number of a device to use (0, 1, 2, 3), or cpu",
                        type=str, default="cpu", required=False, choices=["cpu", "0", "1", "2", "3"])
    parser.add_argument("--env", help="Name of the environment to use i.e. Pong",
                        type=str, required=True, choices=['Pong', 'Qbert', 'Seaquest', 'Space_Invaders', 'Beam_Rider',
                                                          'Road_Runner', 'Ms_Pacman', 'Breakout', 'Asteroids', 'Enduro'])
    parser.add_argument("--extractor", help="Which type of feature extractor to use", type=str,
                        default="lin_concat_ext", required=False,
                        choices=["lin_concat_ext", "fixed_lin_concat_ext",
                                 "cnn_concat_ext", "combine_ext",
                                 "dotproduct_attention_ext", "wsharing_attention_ext",
                                 "reservoir_concat_ext", ])

    parser.add_argument("--debug", type=str, default="False", choices=["True", "False"])
    parser.add_argument("--pi", type=int, nargs='+', default=[256],
                        help="Layers and units for custom actor (pi) network. "
                             "Usage --pi 256 128 for 2 hidden layers with 256 and 128 units respectively.")
    parser.add_argument("--vf", type=int, nargs='+', default=[256],
                        help="Layers and units for custom value function (vf) network. "
                             "Usage --vf 256 128 for 2 hidden layers with 256 and 128 units respectively.")

    parser.add_argument("--fd", type=int, required=False, default=256,
                        help="Fixed size for the skills embedding vector")

    parser.add_argument("--cv", type=int, required=False, default=1,
                        help="Number of convolutional layers to concatenate skills.")

    parser.add_argument("--heads", type=int, required=False, default=4,
                        help="Number of attention heads.")

    parser.add_argument("--ro", type=int, required=False, default=1024,
                        help="Reservoir output size.")

    parser.add_argument("--seed", type=int, required=False,
                        help="Seed to be used for reproducibility. If not provided, a random seed will be used.")

    parser.add_argument("--use-expert", help="if True, loads saved skills models on PPO agent dataset rather then random",
                        type=str, choices=["True", "False"], default="False")

    parser.add_argument("--alg",
                        help="Learning algorithm to use PPO or DQN",
                        type=str, choices=["PPO", "DQN"], default="PPO")

    args = parser.parse_args()

    return args
