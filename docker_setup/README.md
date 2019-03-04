# Prerequisites
This specific image needs a CUDA capable GPU with CUDA version 10.0 and nvidia-docker version 2 installed on the host machine

# Building the image

`docker build -t ffabi/gym:latest .`

# Running the docker container

`nvidia-docker create -p 8192:8192 -p 8193:22 --name ffabi_gym ffabi/gym:latest`

`nvidia-docker start ffabi_gym`

# Attach to the container

`docker exec -it ffabi_gym bash`

# Clone the choosen implemetation of the World Models concept

`git clone https://github.com/AppliedDataSciencePartners/WorldModels.git`

# Run the rest according to this guide:

<https://medium.com/applied-data-science/how-to-build-your-own-world-model-using-python-and-keras-64fb388ba459>
