#ifndef CUDA_IMAGE_PROCESSOR_CUH_
#define CUDA_IMAGE_PROCESSOR_CUH_

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <chrono>
#include <tuple>
#include <stdio.h>
#include <iostream>

__global__ void colorToGreyscaleKernel(unsigned char* input, unsigned char* output, int width, int height);
__global__ void colorToSepiaKernel(unsigned char* input, unsigned char* output, int width, int height);
__global__ void colorToInvertedKernel(unsigned char* input, unsigned char* output, int width, int height);
__global__ void colorToBinaryKernel(unsigned char* input, unsigned char* output, int width, int height);
__global__ void colorToCoolingKernel(unsigned char* input, unsigned char* output, int width, int height);
__global__ void colorToRedBoostKernel(unsigned char* input, unsigned char* output, int width, int height);

__global__ void rotateKernel(unsigned char* input, unsigned char* output, int width, int height);
__global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output, int width, int height, double* kernel, int kernelsize);
__global__ void generateGaussianKernelDevice(double* kernel, int kernelSize, double sigma);

class CudaImageProcessor {
public:
    using KernelFunc = void (*)(unsigned char*, unsigned char*, int, int);

    CudaImageProcessor(cv::Mat& input);
    ~CudaImageProcessor();

    void greyscale();
    void sepia();
    void invert();
    void rotate();
    void binary();
    void cooling();
    void redBoost();
    void blur();

    cv::Mat getOutputImage();
    void processImage(KernelFunc kernel);

    template<typename Func, typename... Args>
    void timeExecution(const std::string& functionName, Func func, Args&&... args) {
        auto start = std::chrono::high_resolution_clock::now();
        (this->*func)(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "Execution time of " << functionName << ": " << elapsed.count() << " ms\n";
    }

private:

private:
    cv::Mat& inputImage;
    cv::Mat outputImage;
    unsigned char *d_input;
    unsigned char *d_output;
    size_t numInputBytes;
    size_t numOutputBytes;
};

#endif 
