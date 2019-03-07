# Prerequisites
This specific image needs a CUDA capable GPU with CUDA version 10.0 and nvidia-docker version 2 installed on the host machine

# Building the image

`git clone https://github.com/ffabi/SemesterProject.git`

`cd SemesterProject/docker_setup`

`docker build -t ffabi/gym:latest .`

# Running the docker container

`mkdir ./ffabi_shared_folder`

`nvidia-docker create -p 8192:8192 -p 8193:22 --name ffabi_gym -v $(pwd)/ffabi_shared_folder:/root/ffabi_shared_folder ffabi/gym:latest`

`nvidia-docker start ffabi_gym`

# Attach to the container

`docker exec -it ffabi_gym bash`

# Clone the implemetation of the World Models concept

`cd ffabi_shared_folder`

`git clone https://github.com/ffabi/SemesterProject.git`

`cd SemesterProject/WorldModels`

# Running AppliedDataSciencePartners - WorldModels

`mkdir data`

`xvfb-run -a -s "-screen 0 1400x900x24" python3 01_generate_data.py car_racing --total_episodes 200 --start_batch 0 --time_steps 300`

`xvfb-run -a -s "-screen 0 1400x900x24" python3 02_train_vae.py --start_batch 0 --max_batch 9 --new_model`

`xvfb-run -a -s "-screen 0 1400x900x24" python3 03_generate_rnn_data.py --start_batch 0 --max_batch 9`

`xvfb-run -a -s "-screen 0 1400x900x24" python3 04_train_rnn.py --start_batch 0 --max_batch 0 --new_model`

`xvfb-run -a -s "-screen 0 1400x900x24" python3 05_train_controller.py car_racing --num_worker 1 --num_worker_trial 2 --num_episode 4 --max_length 1000 --eval_steps 25`



Source:
<https://medium.com/applied-data-science/how-to-build-your-own-world-model-using-python-and-keras-64fb388ba459>


# Running hardmaru - WordlModels


