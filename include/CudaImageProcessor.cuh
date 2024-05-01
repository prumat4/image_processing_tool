#ifndef CUDA_IMAGE_PROCESSOR_CUH_
#define CUDA_IMAGE_PROCESSOR_CUH_

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <chrono>
#include <tuple>
#include <stdio.h>
#include <iostream>

__global__ void colorToGrayscaleKernel(unsigned char* input, unsigned char* output, int width, int height);
__global__ void rotateKernel(unsigned char* input, unsigned char* output, int width, int height);
__global__ void gaussianBlur(unsigned char* input, unsigned char* output, int width, int height, double* kernel, int kernelsize);
__global__ void generateGaussianKernelDevice(double* kernel, int kernelSize, double sigma);

class CudaImageProcessor {
public:
    CudaImageProcessor(cv::Mat& input);
    ~CudaImageProcessor();

    void convertToGreyscale();
    void rotate();
    void blur();
    cv::Mat getOutputImage();

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
