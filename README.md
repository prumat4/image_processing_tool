# image_processing_tool

# Description of the project and motivation
This is a project that implements a high resolution(5000x5000 pixels and more) image processing tool that uses CUDA to process images. Here you can find basic image filters such as grayscale, sepia, binary, etc. The most interesting part was the implementation of the Canny edge detection algorithm and Gaussian blur, here is an article on this topic: https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123?gi=88bcd6109bce. This code is not ready for production, and it was not intended to be. I've always wanted to try interacting with the GPU using C++, but never had the opportunity. Now I do :)

# Project setup:
1. install CUDA, cuDNN and OpencCV
2. there are some useful links at the bottom of the readme
3. 3. build and run:
```bash
bash build.sh && ./imфge_processing_tool [path/to/image]
```

# Results
You can find all resulting images in ```assets``` folder
## Gaussian blur
There is the parameter called ```sigma``` that defines blur intensity
![7](https://github.com/prumat4/image_processing_tool/assets/108902150/01b92cea-ba85-4b23-855c-407b6e0832c6)
![6](https://github.com/prumat4/image_processing_tool/assets/108902150/c72ed9b7-3e6d-4718-87c6-44aa6ea80951)
![5](https://github.com/prumat4/image_processing_tool/assets/108902150/700be7a7-1deb-4917-bc99-f4b9cc989c36)
![4](https://github.com/prumat4/image_processing_tool/assets/108902150/2cd53393-0957-47e2-843c-7a33b06b54b0)


## Canny edge detection
![2](https://github.com/prumat4/image_processing_tool/assets/108902150/3f46a1cf-ca92-4837-b8b8-fa031a2b4e3b)
![1](https://github.com/prumat4/image_processing_tool/assets/108902150/8adaaa49-4080-4fa0-926d-574d5212c184)
![9](https://github.com/prumat4/image_processing_tool/assets/108902150/f5bbe4d3-4ced-4044-8107-7e507a5f7af8)
![8](https://github.com/prumat4/image_processing_tool/assets/108902150/33f7fef2-975c-4b80-8cc6-eaeb9bef1c69)

# Links
cuDNN: https://developer.nvidia.com/cudnn

OpenCv issue: https://github.com/opencv/opencv/issues/24983

cuDNN releases archive: https://developer.nvidia.com/rdp/cudnn-archive

installing guide: https://medium.com/@juancrrn/installing-opencv-4-with-cuda-in-ubuntu-20-04-fde6d6a0a367

link 1: https://github.com/NVlabs/instant-ngp/issues/119#issuecomment-1698809070

link 2:https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#

link 3: https://developer.nvidia.com/cuda-gpus

link 4: https://docs.opencv.org/4.x/d1/dfb/intro.html
