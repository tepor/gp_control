FROM ubuntu:20.04

# 
ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Etc/UTC

# Install debian packages
RUN apt-get update \
    && apt-get install -y \
    git \
    fish \
    python3-pip \
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

# Get Python 3.11 for fun, otherwise comment out
RUN wget -P /tmp https://www.python.org/ftp/python/3.11.0/Python-3.11.0a7.tgz \
    && tar -xzf /tmp/Python-3.11.0a7.tgz -C /tmp \
    && (cd /tmp/Python-3.11.0a7 \
    && ./configure --with-ensurepip=install \
    && make -j 8 \
    && make altinstall)

# Install Python packages (3.8 packages because most don't support 3.11 yet)
RUN pip3 install --no-cache-dir \
    numpy \
    yapf \
    deap \
    pygame \
    gym[classic_control]

# Run fish as the shell because it's nice
CMD fish

