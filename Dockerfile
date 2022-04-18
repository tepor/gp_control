# David Hedger 2022

FROM ubuntu:20.04

# Add some build arguments to help packages install without prompts
ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Etc/UTC

# Install debian packages
RUN apt-get update \
    && apt-get install -y \
    git \
    fish \
    python3-pip \
    # For visualising GP trees
    graphviz \
    graphviz-dev \
    # OpenAI gym dependencies
    swig \
    # For building Python 3.11, otherwise comment out
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    # Clean up the cache for size reasons
    && rm -rf /var/lib/apt/lists/*

# # Get Python 3.11 for fun, otherwise comment out
# RUN wget -P /tmp https://www.python.org/ftp/python/3.11.0/Python-3.11.0a7.tgz \
#     && tar -xzf /tmp/Python-3.11.0a7.tgz -C /tmp \
#     && (cd /tmp/Python-3.11.0a7 \
#     && ./configure --with-ensurepip=install \
#     && make -j 8 \
#     && make altinstall)


# This is broken in a very strange way
# OpenAI gym comes with some default envs built using OpenAI MuJoCo
# The current version of MuJoCo is 2.1
# Versions of MuJoCo <2.0 must be licensed, but >=2.0 are free
# Only older versions of gym(<=0.15.3) support MuJoCo 2.0 (and only up to 2.0)
# You can go and get a free license for MuJoCo 1.5 and that is a possible workaround
# Ultimately way too fiddly and unstable to play with right now

# # Get MuJoCo for physics gyms
# RUN wget -P /tmp https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz \
#     && mkdir ~/.mujoco \
#     && tar -xzf /tmp/mujoco210-linux-x86_64.tar.gz -C ~/.mujoco 

# Install Python packages (3.8 packages because most don't support 3.11 yet)
RUN pip3 install --no-cache-dir \
    numpy \
    pathos \
    yapf \
    deap \
    pygraphviz \
    pygame \
    gym[classic_control,box2d]

# Run fish as the shell because it's nice
CMD fish

