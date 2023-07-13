FROM nvidia/cudagl:11.1-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt update && apt install -y git python3 python3-pip vim cmake ffmpeg

# Make image smaller by not caching downloaded pip pkgs
ARG PIP_NO_CACHE_DIR=1

# Install pytorch for example, and ensure sim works with all our required pkgs
ARG TORCH=1.10.0
ARG CUDA=cu111
# Pytorch and torch_geometric w/ deps
RUN pip3 install torch==${TORCH}+${CUDA} \
    -f https://download.pytorch.org/whl/torch_stable.html
    
#RUN pip3 install torch-scatter==2.0.8 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.2 torch-geometric==1.7.2

RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.10.0+cu111.html
#RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.10.0+cu111.html
RUN pip3 install torch-sparse==0.6.13
RUN pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.10.0+cu111.html
RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.10.0+cu111.html
RUN pip install torch-geometric==2.0
RUN pip install -U tensorboardx



# pytorch_geometric can be a bit buggy during install
RUN python3 -c "import torch; print(torch.__version__)"
RUN python3 -c "import torch_geometric"

ADD ./src/rllib_multi_agent_demo/requirements.txt \
    /build/requirements/requirements_demo.txt
ADD ./requirements.txt \
    /build/requirements/requirements_app.txt

RUN pip3 install \
    -r /build/requirements/requirements_demo.txt \
    -r /build/requirements/requirements_app.txt

RUN pip install protobuf==3.20
# Make PyGame render in headless mode
ENV SDL_VIDEODRIVER dummy
ENV SDL_AUDIODRIVER dsp

WORKDIR /home
