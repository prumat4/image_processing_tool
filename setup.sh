#!/bin/bash

sudo apt update -y

sudo apt install -y build-essential cmake git unzip pkg-config \
libjpeg-dev libpng-dev libtiff-dev \
libavcodec-dev libavformat-dev libswscale-dev \
libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev \
libatlas-base-dev gfortran python3-dev python3-numpy \
libtbb2 libtbb-dev libdc1394-22-dev

# Optional: Install CUDA if not already installed
# Note: It is recommended to manually install CUDA following the instructions
# from NVIDIA to ensure compatibility with your specific GPU and drivers

# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# wget http://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda-repo-ubuntu2004-11-2-local_11.2.2-460.32.03-1_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu2004-11-2-local_11.2.2-460.32.03-1_amd64.deb
# sudo apt-key add /var/cuda-repo-ubuntu2004-11-2-local/7fa2af80.pub
# sudo apt-get update
# sudo apt-get -y install cuda

mkdir -p ~/opencv_build && cd ~/opencv_build

git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

cd opencv
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D WITH_TBB=ON \
      -D WITH_CUDA=ON \
      -D BUILD_EXAMPLES=ON \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules ..

make -j$(nproc)

sudo make install

chmod +x build.sh
