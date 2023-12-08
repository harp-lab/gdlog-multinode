# use CUDA 11.8 because btree need it
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Install required Ubuntu packages
RUN apt-get clean && apt-get update -y -qq
RUN apt-get install -y wget git build-essential curl cmake vim
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y graphviz python-is-python3

# use bash as default shell
SHELL ["/bin/bash", "-c"]

# setup micromamba (a faster conda)
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
RUN ./bin/micromamba shell init -s bash -p ~/micromamba
RUN source ~/.bashrc
RUN micromamba create -n gdlog matplotlib pandas scipy -c conda-forge
# SHELL ["micromamba", "run", "-n", "gdlog", "/bin/bash", "-c"]
# optional: oh-my-bash
RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh)"

COPY . /gdlog
