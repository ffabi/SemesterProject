FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
MAINTAINER ffabi

RUN apt update && apt install -y python3.6 python3-pip python3-dev
RUN pip3 install --upgrade pip

RUN pip3 install tensorflow-gpu keras numpy

#RUN pip3 install torchvision_nightly
#RUN pip3 install torch_nightly -f https://download.pytorch.org/whl/nightly/cu100/torch_nightly.html

RUN pip3 install gym
RUN pip3 install gym[box2d]
RUN pip3 install gym[atari]

RUN pip3 install jupyterlab
RUN apt install -y python3-opengl xvfb fontconfig

COPY jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
COPY jupyter_notebook_config.json /root/.jupyter/jupyter_notebook_config.json

RUN apt install -y libopenmpi-dev ssh sudo nano
RUN pip3 install mpi4py pympler cma

RUN echo "PermitRootLogin yes\nSubsystem sftp internal-sftp" > /etc/ssh/sshd_config
RUN echo "root:init" | chpasswd

WORKDIR /root
EXPOSE 8192

CMD service ssh start && jupyter notebook --allow-root