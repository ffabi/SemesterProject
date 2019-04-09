# python 03_generate_rnn_data.py

from vae.arch import VAE
import argparse
import config
import numpy as np


def generate(args):
    num_files = args.num_files
    validation = args.validation

    vae = VAE()

    weights = "final.h5"

    try:
        vae.set_weights("./vae/" + weights)
    except:
        print("./vae/" + weights + " does not exist")
        raise FileNotFoundError

    for file_id in range(num_files):
        first_item = True
        print("Generating file {}...".format(file_id))

        for env_name in config.train_envs:

            try:
                if validation:
                    obs_data = np.load("./data/obs_valid_" + env_name + ".npz")["arr_0"]
                    action_data = np.load("./data/action_valid_" + env_name + ".npz")["arr_0"]
                else:
                    new_obs_data = np.load("./data/obs_data_" + env_name + "_" + str(file_id) + ".npz")["arr_0"]
                    new_action_data = np.load("./data/action_data_" + env_name + "_" + str(file_id) + ".npz")["arr_0"]
                    if first_item:
                        obs_data = new_obs_data
                        action_data = new_action_data
                        first_item = False
                    else:
                        obs_data = np.concatenate([obs_data, new_obs_data])
                        action_data = np.concatenate([action_data, new_action_data])
                    print("Found {}...current data size = {} episodes".format(env_name, len(obs_data)))
            except:
                pass

        if validation:
            rnn_input, rnn_output = vae.generate_rnn_data(obs_data, action_data)
            np.savez_compressed("./data/rnn_input_" + env_name + "_valid", rnn_input)
            np.savez_compressed("./data/rnn_output_" + env_name + "_valid", rnn_output)
            break
        if not first_item:
            rnn_input, rnn_output = vae.generate_rnn_data(obs_data, action_data)
            np.savez_compressed("./data/rnn_input_" + env_name + "_" + str(file_id), rnn_input)
            np.savez_compressed("./data/rnn_output_" + env_name + "_" + str(file_id), rnn_output)
        else:
            print("no data found for batch number {}".format(file_id))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data for the RNN")
    parser.add_argument("--num_files", type=int, default=10, help="The number of files to be analyzed")
    parser.add_argument("--validation", action="store_true", help="Save to rnn_input_valid.npz")

    args = parser.parse_args()

    generate(args)
