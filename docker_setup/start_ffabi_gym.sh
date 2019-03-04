#!/bin/bash

nvidia-docker create -p 8192:8192 -p 8193:22 --name ffabi_gym --rm -v /home/docker_home:/root/docker_home ffabi/gym:latest

nvidia-docker start ffabi_gym