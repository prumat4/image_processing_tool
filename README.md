# image_processing_tool

# To do: 
- [ ] fix cmake config and adjust readme
- [ ] 
- [ ] 
- [ ] 
- [ ] 



# Project setup:
Note: steps [2 - n] can be performed using setup.sh
1. install cuda: https://developer.nvidia.com/cuda-downloads

2. add CUDA to your PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

3. install dependecies:
```bash
sudo apt update
sudo apt install build-essential cmake git unzip pkg-config libjpeg-dev libpng-dev libtiff-dev \
libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev \
libatlas-base-dev gfortran python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-22-dev
```

4. install OpenCV
```bash
sudo apt-get install -y libopencv-dev
```

