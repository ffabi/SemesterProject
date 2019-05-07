FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
MAINTAINER ffabi

RUN apt update --fix-missing
RUN apt install -y python3.5 python3-pip python3-dev
RUN apt install -y libopenmpi-dev ssh sudo nano gcc cmake git
RUN apt install -y build-essential freeglut3 freeglut3-dev libxi-dev libxmu-dev zlib1g-dev

RUN pip3 install --upgrade pip
RUN pip3 install mpi4py pympler cma
RUN pip3 install tensorflow-gpu==1.13.1 keras

#RUN pip3 install torchvision_nightly
#RUN pip3 install torch_nightly -f https://download.pytorch.org/whl/nightly/cu100/torch_nightly.html

RUN pip3 install gym
RUN pip3 install gym[box2d]
RUN pip3 install gym[atari]

RUN pip3 install jupyterlab
RUN apt install -y python3-opengl xvfb fontconfig

COPY jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
COPY jupyter_notebook_config.json /root/.jupyter/jupyter_notebook_config.json


RUN echo "PermitRootLogin yes\nSubsystem sftp internal-sftp" > /etc/ssh/sshd_config
RUN echo "root:init" | chpasswd

RUN pip3 install --upgrade tornado==5.1.1
RUN pip3 install matplotlib
RUN pip3 install numpy==1.16.1 # 1.16.3 has a bug now, "Object arrays cannot be loaded when allow_pickle=False" is the symptom

RUN apt install -y screen

WORKDIR /root

VOLUME /ffabi_shared_folder

CMD service ssh start && jupyter notebook --allow-root
