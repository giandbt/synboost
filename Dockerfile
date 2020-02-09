## version = v1.0
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git curl vim ca-certificates \
                                                                 libjpeg-dev libpng-dev libsm6 libxext6 libxrender1 libfontconfig1 graphviz &&\
                                                                 rm -rf /var/lib/apt/lists/*
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh

RUN /opt/conda/bin/conda init bash
RUN /opt/conda/bin/conda create --name uncertainty
RUN echo "source activate uncertainty" > ~/.bashrc
ENV PATH /opt/conda/bin:$PATH

RUN pip install \
    torch==1.4.0 torchvision==0.5.0 \
    dominate==2.4.0 \
    dill \
    scipy \
    pandas \
    Pillow==7.0.0

WORKDIR /home/giancarlo/