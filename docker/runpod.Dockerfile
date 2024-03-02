FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Install System Packages
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git             \
    nano            \
    nginx           \
    tzdata          \
    expect          \
    ca-certificates \
    openssh-server  \
    build-essential \
    python3.10-dev python3.10-venv \
    gnome-keyring wget curl ca-certificates \
    lsb-release mpich vim \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*


# Set up Python and pip
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py

RUN pip install --upgrade pip && pip install jupyterlab


RUN apt-get install -y ca-certificates gpg wget 
RUN test -f /usr/share/doc/kitware-archive-keyring/copyright || wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
RUN echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null
RUN apt-get update && apt-get install -y kitware-archive-keyring
RUN apt-get update && apt-get install -y cmake

# expose port
EXPOSE 8000

# NGINX Proxy
RUN wget -O init-deb.sh https://www.linode.com/docs/assets/660-init-deb.sh && \
    mv init-deb.sh /etc/init.d/nginx && \
    chmod +x /etc/init.d/nginx && \
    /usr/sbin/update-rc.d -f nginx defaults


COPY ./script/runpod/nginx.conf /etc/nginx/nginx.conf
COPY ./script/runpod/nginx-readme /usr/share/nginx/html/readme.html

# Copy the README.md
COPY ./README.md /usr/share/nginx/html/README.md

# Start Scripts
COPY ./script/runpod/start.sh /start.sh
RUN chmod +x /start.sh

# use gcc-13 and g++-13 on ubuntu 22.04
# SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# ARG USER=gdlog
# ARG PASS="gdlog"
# RUN useradd -m -s /bin/bash $USER && echo "$USER:$PASS" | chpasswd
# USER gdlog

RUN apt-get install -y openmpi-bin libopenmpi-dev

COPY --chown=gdlog:gdlog . /opt/gdlog
WORKDIR /opt/gdlog

RUN rm -r build
RUN cmake -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -Bbuild . && cd build && make -j
RUN chmod -R 757 /opt/gdlog

CMD [ "/start.sh" ]
