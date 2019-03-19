#xvfb-run -a -s "-screen 0 1400x900x24" python3 02_train_vae.py --start_batch 0 --max_batch $(ls data | grep obs | wc -l) --new_model

from vae.arch import VAE
import argparse
import numpy as np
import config

def main(args):

    max_batch = args.max_batch
    new_model = args.new_model

    vae = VAE()

    if not new_model:
        try:
            vae.set_weights('./vae/weights.h5')
        except:
            print("Either set --new_model or ensure ./vae/weights.h5 exists")
            raise

    
    vae.train(max_batch)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= 'Train VAE')
    parser.add_argument('--max_batch', type=int, default = 10, help='The max batch number') # --max_batch $(ls data | grep obs | wc -l)
    parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
    args = parser.parse_args()

    main(args)
