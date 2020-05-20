FROM ubuntu:18.04 AS garage-18.04

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

# apt dependencies
RUN \
  apt-get -y -q update && \
  # Prevents debconf from prompting for user input
  # See https://github.com/phusion/baseimage-docker/issues/58
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    # Dockerfile deps
    wget \
    unzip \
    git \
    curl \
    # For building glfw
    cmake \
    xorg-dev \
    # mujoco_py
    # See https://github.com/openai/mujoco-py/blob/master/Dockerfile
    # 18.04 repo is old, install glfw from source instead
    # libglfw3 \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    # OpenAI baselines
    libopenmpi-dev \
    # virtualenv
    python3 \
    python3-pip \
    python3-tk \
    python3-virtualenv && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Build GLFW because the Ubuntu 18.04 version is too old
# See https://github.com/glfw/glfw/issues/1004
RUN apt-get purge -y -v libglfw* && \
  wget https://github.com/glfw/glfw/releases/download/3.3/glfw-3.3.zip && \
  unzip glfw-3.3.zip && \
  rm glfw-3.3.zip && \
  cd glfw-3.3 && \
  mkdir glfw-build && \
  cd glfw-build && \
  cmake -DBUILD_SHARED_LIBS=ON -DGLFW_BUILD_EXAMPLES=OFF -DGLFW_BUILD_TESTS=OFF -DGLFW_BUILD_DOCS=OFF .. && \
  make -j"$(nproc)" && \
  make install && \
  cd ../../ && \
  rm -rf glfw

RUN groupadd -g 999 garage-user && \
    useradd -m -r -u 999 -g garage-user garage-user && \
    chown -R garage-user:garage-user /home/garage-user
USER garage-user
ENV HOME /home/garage-user
WORKDIR $HOME

# MuJoCo 2.0 (for dm_control)
RUN mkdir -p $HOME/.mujoco && \
  wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip && \
  unzip mujoco.zip -d $HOME/.mujoco && \
  rm mujoco.zip && \
  ln -s $HOME/.mujoco/mujoco200_linux $HOME/.mujoco/mujoco200
  ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin

# Create virtualenv
ENV VIRTUAL_ENV=$HOME/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Prevent pip from complaining about available upgrades
RUN pip install --upgrade pip

# We need a MuJoCo key to install mujoco_py
# In this step only the presence of the file mjkey.txt is required, so we only
# create an empty file
RUN touch $HOME/.mujoco/mjkey.txt && \
  pip install garage[all]==2020.5rc1 && \
  rm -r $HOME/.cache/pip && \
  rm $HOME/.mujoco/mjkey.txt


# Headless machine with xvfb, i.e. so rendering works
FROM garage-18.04 AS garage-headless-18.04
USER root
RUN \
  apt-get -y -q update && \
  # Prevents debconf from prompting for user input
  # See https://github.com/phusion/baseimage-docker/issues/58
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    # Dummy X server
    xvfb \
    pulseaudio && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*
USER garage-user

# Ready, set, go.
ENTRYPOINT ["docker/entrypoint-headless.sh"]


# Special setup for garage developers, which installs garage from source
# Assumes $PWD is the garage repository.
FROM garage-headless-18.04 AS garage-dev-18.04
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy over just setup.py first, so the Docker cache doesn't expire until
# dependencies change
#
# Files needed to run setup.py
# - README.md
# - VERSION
# - scripts/garage
# - src/garage/__init__.py
# - setup.py
COPY --chown=garage-user:garage-user README.md $HOME/code/garage/README.md
COPY --chown=garage-user:garage-user VERSION $HOME/code/garage/VERSION
COPY --chown=garage-user:garage-user scripts/garage $HOME/code/garage/scripts/garage
COPY --chown=garage-user:garage-user src/garage/__init__.py $HOME/code/garage/src/garage/__init__.py
COPY --chown=garage-user:garage-user setup.py $HOME/code/garage/setup.py
WORKDIR $HOME/code/garage

# Prevent pip from complaining about available upgrades
RUN pip install --upgrade pip

# Pre-build pre-commit env
COPY --chown=garage-user:garage-user .pre-commit-config.yaml $HOME/code/garage
RUN ls -la && echo "$(id -u)" && ls -la $HOME
RUN git init && \
  pip install pre-commit && \
  pre-commit install && \
  pre-commit install-hooks

# Install deps (but not the code)
# Note that we don't delete the pip cache for dev builds
RUN touch $HOME/.mujoco/mjkey.txt && \
  pip install .[all,dev] && \
  rm $HOME/.mujoco/mjkey.txt

# Add code stub last (ensures code changes have the shortest builds)
COPY --chown=garage-user:garage-user . $HOME/code/garage

# Build and install the sdist
RUN python3 setup.py sdist && \
    cp dist/*.tar.gz dist/garage.tar.gz && \
    pip install dist/garage.tar.gz[all,dev] && \
    cd benchmarks && python3 setup.py sdist && \
    cp dist/*.tar.gz dist/benchmarks.tar.gz && \
    pip install dist/benchmarks.tar.gz

# Test target
FROM garage-dev-18.04 AS garage-test-18.04
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
CMD nice -n 11 pytest -v -m 'not huge and not flaky' --durations=20
