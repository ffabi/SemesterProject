#python 04_train_rnn.py --new_model

from rnn.arch import RNN
import argparse
import numpy as np

def main(args):
    num_files = args.num_files
    load_model = args.load_model

    rnn = RNN()

    if not load_model=="None":
        try:
            print("Loading model " + load_model)
            rnn.set_weights(load_model)
        except:
            print("Either don't set --load_model or ensure " + load_model + " exists")
            raise

    rnn.train(num_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RNN')
    parser.add_argument('--num_files', type=int, default = 10, help='The number of files')
    parser.add_argument('--load_model', type=str, default = "None", help='load an existing model')
    args = parser.parse_args()

    main(args)
