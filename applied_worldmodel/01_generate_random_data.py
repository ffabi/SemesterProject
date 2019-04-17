# xvfb-run -s "-screen 0 1400x900x24" python generate_data.py car_racing --total_episodes 200 --start_batch 0 --time_steps 300

import numpy as np
import random
import config
# from sklearn.utils import shuffle
# import matplotlib.pyplot as plt

from env import make_env

import argparse


def main(args):
    env_name = args.env_name
    total_episodes = args.total_episodes
    start_batch = args.start_batch
    time_steps = args.time_steps
    render = args.render
    file_size = args.file_size
    run_all_envs = args.run_all_envs
    validation = args.validation
    start_frame = args.start_frame

    if validation:
        total_episodes = file_size

    if run_all_envs:
        envs_to_generate = config.train_envs
    else:
        envs_to_generate = [env_name]

    for current_env_name in envs_to_generate:
        print("Generating data for env {}".format(current_env_name))

        env = make_env(current_env_name)
        s = 0
        batch = start_batch

        batch_size = min(file_size, total_episodes)

        while s < total_episodes:
            obs_data = []
            action_data = []

            for i_episode in range(batch_size):
                print("-----")
                observation = env.reset()
                observation = config.adjust_obs(observation)

                # essential for saving as well
                env.render()

                done = False
                action = env.action_space.sample()
                time = 0
                obs_sequence = []
                action_sequence = []

                while time < time_steps and not done:
                    time = time + 1

                    action = config.generate_data_action(time, action)

                    observation, reward, done, info = env.step(action)
                    observation = config.adjust_obs(observation)

                    if time > start_frame:
                        obs_sequence.append(observation)  # [:56]?
                        action_sequence.append(action)

                    if render:
                        env.render()

                obs_data.append(obs_sequence)
                action_data.append(action_sequence)

                print("File {} Episode {} finished after {} timesteps".format(batch, i_episode, time + 1))
                print("Current dataset contains {} observations".format(sum(map(len, obs_data))))

                s = s + 1

            print("Saving dataset for batch {}".format(batch))

            if validation:
                np.savez_compressed("./data/obs_valid_" + current_env_name, obs_data)
                np.savez_compressed("./data/action_valid_" + current_env_name, action_data)
            else:
                # np.random.shuffle(obs_data)

                # obs_data, action_data = shuffle(obs_data, action_data)

                np.savez_compressed("./data/obs_data_" + current_env_name + "_" + str(batch), obs_data)
                np.savez_compressed("./data/action_data_" + current_env_name + "_" + str(batch), action_data)

            batch = batch + 1

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create new training data")
    parser.add_argument("--env_name", default="car_racing", type=str, help="name of environment")
    parser.add_argument("--total_episodes", type=int, default=640, help="total number of episodes to generate")
    parser.add_argument("--start_batch", type=int, default=0, help="start_batch number")
    parser.add_argument("--time_steps", type=int, default=230, help="how many timesteps at start of episode?")
    parser.add_argument("--start_frame", type=int, default=30, help="how many timesteps at start of episode?")
    parser.add_argument("--render", action="store_true", help="render the env as data is generated")
    parser.add_argument("--file_size", type=int, default=64, help="how many episodes in a batch (one file)")
    parser.add_argument("--run_all_envs", action="store_true",
                        help="if true, will ignore env_name and loop over all envs in train_envs variables in config.py")
    parser.add_argument("--validation", action="store_true",
                        help="save to obs_valid.npz")

    args = parser.parse_args()
    main(args)
