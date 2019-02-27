ARG cuda_version=9.0
ARG cudnn_version=7
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

# Install conda
RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    apt-get update && \
    apt-get install -y wget git libhdf5-dev g++ graphviz openmpi-bin && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo "c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a *Miniconda3-4.2.12-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-4.2.12-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.2.12-Linux-x86_64.sh

# Install python dependencies
RUN conda install -y python=3.6 && \
    pip install --upgrade pip && \
    pip install cmake==3.13.0 && \
    conda install -c anaconda tensorflow-gpu=1.8.0 && \
    conda install -c conda-forge tensorboard=1.8.0 && \
    conda install Pillow scikit-learn notebook pandas matplotlib mkl nose pyyaml six h5py && \
    conda install theano pygpu bcolz && \
    conda clean -yt

ADD requirements.txt /home/opt/
RUN pip install -r /home/opt/requirements.txt
ENV PYTHONPATH='/src/:$PYTHONPATH'

# Install packages
RUN apt-get install -qqy x11-apps
RUN apt-get install -y wget vim

# Minimize image size 
RUN (apt-get autoremove -y; \
apt-get autoclean -y)

# Set up work dir
WORKDIR /src
COPY files/ /src/files/
COPY models/ /src/models/
COPY scripts/ /src/scripts/
COPY main.py /src/main.py
COPY options.json /src/options.json

EXPOSE 8888 6006

CMD /bin/bash
