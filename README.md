- [Prerequisites](#prerequisites)
- [Pulling the image](#pulling-the-image)
- [Building the image](#building-the-image)
- [Running the docker container](#running-the-docker-container)
- [Attach to the container](#attach-to-the-container)
- [Clone the implemetation of the World Models concept](#clone-the-implemetation-of-the-world-models-concept)
- [Running AppliedDataSciencePartners - WorldModels](#running-applied-data-science-partners-world-models)
- [Running hardmaru - WordlModels](#running-hardmaru-wordl-models)

# Prerequisites
This specific image needs a CUDA capable GPU with CUDA version 10.0 and nvidia-docker installed on the host machine

# Create working directory and clone the repo
`mkdir ./ffabi_shared_folder`

`cd ffabi_shared_folder`

`git clone https://github.com/ffabi/SemesterProject.git`

# Pulling the image
`docker pull ffabi/gym:10`

or:
# Building the image
`cd SemesterProject/docker_setup`
`docker build -f Dockerfile -t ffabi/gym:10 .`
# Running the docker container

`nvidia-docker create -p 8192:8192 -p 8193:22 -p 8194:8194 --name ffabi_gym -v $(pwd)/ffabi_shared_folder:/root/ffabi_shared_folder ffabi/gym:10`

`nvidia-docker start ffabi_gym`

# Attach to the container
`docker exec -it ffabi_gym bash`

# Running AppliedDataSciencePartners - WorldModels
`mkdir data`

`xvfb-run -a python3 01_generate_random_data.py --total_episodes 640 --file_size 64 --start_batch 0`

`xvfb-run -a python3 02_train_vae.py --start_batch 0 --max_batch 9 --new_model`

`xvfb-run -a python3 03_generate_rnn_data.py --start_batch 0 --max_batch 9`

`xvfb-run -a python3 04_train_rnn.py --start_batch 0 --max_batch 0 --new_model`

`xvfb-run -a python3 05_train_controller.py car_racing --num_worker 1 --num_worker_trial 2 --num_episode 4 --max_length 1000 --eval_steps 25`

Source:
<https://medium.com/applied-data-science/how-to-build-your-own-world-model-using-python-and-keras-64fb388ba459>


