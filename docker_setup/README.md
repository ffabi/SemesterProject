# Building image

`docker build -t ffabi/gym:latest .`

# Running the docker container

`nvidia-docker create -p 8192:8192 -p 8193:22 --name ffabi_gym ffabi/gym:latest`

`nvidia-docker start ffabi_gym`

# Attach to the container

`docker attach ffabi_gym`