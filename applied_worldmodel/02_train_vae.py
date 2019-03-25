#xvfb-run -a -s "-screen 0 1400x900x24" python3 02_train_vae.py --start_batch 0 --max_batch $(ls data | grep obs | wc -l) --new_model

from vae.arch import VAE
import argparse
import numpy as np
import config

def main(args):

    max_batch = args.max_batch
    load_model = args.load_model

    vae = VAE()

    if not load_model=="None":
        try:
            print("Loading model " + load_model)
            vae.set_weights(load_model)
        except:
            print("Either don't set --load_model or ensure " + load_model + " exists")
            raise

    
    vae.train(max_batch)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= 'Train VAE')
    parser.add_argument('--max_batch', type=int, default = 10, help='The max batch number') # --max_batch $(ls data | grep obs | wc -l)
    parser.add_argument('--load_model', type=str, default = "None", help='load an existing model')
    args = parser.parse_args()

    main(args)
