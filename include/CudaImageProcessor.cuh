#ifndef CUDA_IMAGE_PROCESSOR_CUH_
#define CUDA_IMAGE_PROCESSOR_CUH_

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

__global__ void colorToGrayscaleKernel(unsigned char* input, unsigned char* output, int width, int height);

class CudaImageProcessor {
public:
    CudaImageProcessor(cv::Mat& input);
    ~CudaImageProcessor();

    void convertToGreyscale();
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
    cv::Mat& inputImage;
    cv::Mat outputImage;
    unsigned char *d_input, *d_output;
    size_t numInputBytes, numOutputBytes;
};

#endif 
