- [Prerequisites](#prerequisites)
- [Pulling the image](#pulling-the-image)
- [Building the image](#building-the-image)
- [Running the docker container](#running-the-docker-container)
- [Attach to the container](#attach-to-the-container)
- [Clone the implemetation of the World Models concept](#clone-the-implemetation-of-the-world-models-concept)
- [Running AppliedDataSciencePartners - WorldModels](#running-applied-data-science-partners-world-models)
- [Running hardmaru - WordlModels](#running-hardmaru-wordl-models)

# Prerequisites
This specific image needs a CUDA capable GPU with CUDA version 10.0 and nvidia-docker version 2 installed on the host machine

# Reproducing the results

   ### Create working directory and clone the repo

1. `mkdir ffabi_shared_folder`

1. `cd ffabi_shared_folder`

1. `git clone https://github.com/ffabi/SemesterProject.git`

   ### Building the image:

1. `cd SemesterProject/docker_setup`

1. `docker build -f Dockerfile -t ffabi/gym:10 .`

1. `cd ../../..`

    ### Running the docker container:

1. `nvidia-docker create -p 8192:8192 -p 8193:22 -p 8194:8194 --name ffabi_gym -v $(pwd)/ffabi_shared_folder:/root/ffabi_shared_folder ffabi/gym:10`

1. `nvidia-docker start ffabi_gym`

    ### Attach to the container:

1. `docker exec -it ffabi_gym bash`

    ### Running AppliedDataSciencePartners - WorldModels:

1. `cd ffabi_shared_folder/SemesterProject/applied_worldmodel`

1. `mkdir data log`

    ### Generate random rollouts:
    I recommend to use screen:
    
1. `screen -R train` to enter screen 'train' (press Ctrl+A and Ctrl+D to exit)
 
1. `xvfb-run -a python3 01_generate_random_data.py`

1. `xvfb-run -a python3 01_generate_random_data.py --validation`

    ### Train the VAE:

1. `xvfb-run -a python3 02_train_vae.py --num_files 10`

   ### Generate input for the RNN:

1. `xvfb-run -a python3 03_generate_rnn_data.py --start_batch 0 --max_batch 9`

   ### Train the RNN:

1. `xvfb-run -a python3 04_train_rnn.py --start_batch 0 --max_batch 0 --new_model`

   ### Train the Controller:

1. `xvfb-run -a python3 05_train_controller.py car_racing --num_worker 1 --num_worker_trial 2 --num_episode 4 --max_length 1000 --eval_steps 25`

Source:
<https://medium.com/applied-data-science/how-to-build-your-own-world-model-using-python-and-keras-64fb388ba459>


