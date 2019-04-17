# python 03_generate_rnn_data.py
# TODO: organize code
from vae.arch import VAE
import argparse
import config
import numpy as np


def generate(args):
    num_files = args.num_files
    validation = args.validation
    seq_len = args.seq_len

    vae = VAE()

    weights = "final.h5"

    try:
        vae.set_weights("./vae/" + weights)
    except:
        print("./vae/" + weights + " does not exist")
        raise FileNotFoundError

    for file_id in range(num_files):
        print("Generating file {}...".format(file_id))

        obs_data = []
        action_data = []

        for env_name in config.train_envs:
            try:
                if validation:
                    new_obs_data = np.load("./data/obs_valid_" + env_name + ".npz")["arr_0"]
                    new_action_data = np.load("./data/action_valid_" + env_name + ".npz")["arr_0"]
                else:
                    new_obs_data = np.load("./data/obs_data_" + env_name + "_" + str(file_id) + ".npz")["arr_0"]
                    new_action_data = np.load("./data/action_data_" + env_name + "_" + str(file_id) + ".npz")["arr_0"]

                index = 0
                for episode in new_obs_data:
                    # print(len(episode))
                    if len(episode) != seq_len:
                        new_obs_data = np.delete(new_obs_data, index)
                        new_action_data = np.delete(new_action_data, index)
                    else:
                        index += 1

                obs_data = np.append(obs_data, new_obs_data)
                action_data = np.append(action_data, new_action_data)

                print("Found {}...current data size = {} episodes".format(env_name, len(obs_data)))
            except Exception as e:
                print(e)
                pass

            if validation:
                rnn_input, rnn_output = vae.generate_rnn_data(obs_data, action_data)
                np.savez_compressed("./data/rnn_input_" + env_name + "_valid", rnn_input)
                np.savez_compressed("./data/rnn_output_" + env_name + "_valid", rnn_output)
            else:
                rnn_input, rnn_output = vae.generate_rnn_data(obs_data, action_data)
                np.savez_compressed("./data/rnn_input_" + env_name + "_" + str(file_id), rnn_input)
                np.savez_compressed("./data/rnn_output_" + env_name + "_" + str(file_id), rnn_output)

        if validation:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data for the RNN")
    parser.add_argument("--num_files", type=int, default=10, help="The number of files to be analyzed")
    parser.add_argument("--validation", action="store_true", help="Save to rnn_input_valid.npz")
    parser.add_argument("--seq_len", type=int, default=200, help="The number of files to be analyzed")

    args = parser.parse_args()

    generate(args)
