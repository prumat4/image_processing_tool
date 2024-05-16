# image_processing_tool

# To do: 
- [x] convert to greyscale and mb other filters too
- [x] image rotation
- [ ] image scaling
- [x] Canny edge detector
- [ ] hdr
- [ ] Depth Map Generation
- [ ] noise reduction
- [ ] image filtering
- [x] blur
- [ ] play around with params to get best performance for my gpu for each method, maybe create some .json file to parse this params, so user can easily change them without digging the code 
- [ ] fix cmake config and adjust readme
- [ ] add separate file, smth like performance.cpp that will run operation from CudaImageProcessor and similar operation from opencv to compare time execution
there should be some kinda fkag to run the actual perforamnce test (lot of images for each operation for best accuracy) or visual test, where u will see the cuda and opencv image processing results simultaneously
- [ ] create diagrams with comparison of performance both for CPU and GPU, use ROOT library https://root.cern/
- [ ] write some unit tests and make them auto run in the github
- [ ] in assets folder there should be only input images, and there must be some function, which generates greyscale, blur and so on for assets folder ... 
- [ ] add proper description for this repo, that will answer this question: Why I wanted to implement this? Where can I see the results? 
- [ ] VIDEO PROCESSING TOOL???????

# Project setup:
Note: steps [2 - n] can be performed using setup.sh, do ```chmod +x setup.sh```
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

4. install cuDNN: https://developer.nvidia.com/cudnn (version 8.9.7, because there is this issue with cuda 9.1.1 https://github.com/opencv/opencv/issues/24983)
![Screenshot from 2024-05-17 00-00-51](https://github.com/prumat4/image_processing_tool/assets/108902150/214912e0-158e-430f-ba87-b0f8fe3ac155)

```
try to uninstall 9.1.1 and install 8.9.7 if needed
```
archieve releases: https://developer.nvidia.com/rdp/cudnn-archive
5. How install OpenCV with CUDA: https://medium.com/@juancrrn/installing-opencv-4-with-cuda-in-ubuntu-20-04-fde6d6a0a367
I faced this issue: https://github.com/NVlabs/instant-ngp/issues/119, this suggestion worked for me: https://github.com/NVlabs/instant-ngp/issues/119#issuecomment-1698809070
<!-- ```bash
sudo apt-get install -y libopencv-dev
``` -->

# Links: 
1. CUDA docs: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#
2. GPU compute capability: https://developer.nvidia.com/cuda-gpus, mine is NVIDIA GeForce RTX 3060 Laptop GPU with capability of 8.6 => 
3. OpenCV docs: https://docs.opencv.org/4.x/d1/dfb/intro.html

