# image_processing_tool

# Description of the project and motivation
This is a project that implements an image processing tool that uses CUDA to process images. Here you can find basic image filters such as grayscale, sepia, binary, etc. The most interesting part was the implementation of the Canny edge detection algorithm and Gaussian blur, here is an article on this topic: https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123?gi=88bcd6109bce. This code is not ready for production, and it was not intended to be. I've always wanted to try interacting with the GPU using C++, but never had the opportunity. Now I do :)

# Project setup:
1. Install CUDA, cuDNN and OpencCV
2. here is some links that might be useful while setting up the project:
https://developer.nvidia.com/cudnn
https://github.com/opencv/opencv/issues/24983
https://developer.nvidia.com/rdp/cudnn-archive
https://medium.com/@juancrrn/installing-opencv-4-with-cuda-in-ubuntu-20-04-fde6d6a0a367
https://github.com/NVlabs/instant-ngp/issues/119#issuecomment-1698809070
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#
https://developer.nvidia.com/cuda-gpus
https://docs.opencv.org/4.x/d1/dfb/intro.html
3. build and run:
```bash
bash build.sh && ./iamge_processing_tool [path/to/image]
```

# Results
