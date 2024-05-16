#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include "CudaImageProcessor.cuh"

namespace fs = std::filesystem;

void compareBlur(cv::Mat& inputImage, double sigma) {
    CudaImageProcessor processor(inputImage);
    cv::imshow("Input image", inputImage);

    auto GPUStart = std::chrono::high_resolution_clock::now();
    processor.blur(sigma);
    auto GPUEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> GPUElapsed = GPUEnd - GPUStart;
    std::cout << "Execution time of custom GPU blur, with sigma = " << sigma << " is: " << GPUElapsed.count() << " ms\n";
    
    cv::Mat GPUImage = processor.getOutputImage();
    cv::imshow("Custom GPU image", GPUImage);

    cv::cuda::GpuMat d_inputImage, d_outputImage;
    d_inputImage.upload(inputImage);

    double kernelSize = 6 * sigma + 1;
    auto OpenCV_GPUStart = std::chrono::high_resolution_clock::now();
    cv::cuda::GaussianBlur(d_inputImage, d_outputImage, cv::Size(kernelSize, kernelSize), 0);
    auto OpenCV_GPUEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> OpenCV_GPUElapsed = OpenCV_GPUEnd - OpenCV_GPUStart;
    std::cout << "Execution time of OpenCV GPU blur, with sigma = " << sigma << " is: " << OpenCV_GPUElapsed.count() << " ms\n";

    cv::Mat OpenCV_GPUImage;
    d_outputImage.download(OpenCV_GPUImage);
    cv::imshow("OpenCV GPU image", OpenCV_GPUImage);

    cv::Mat CPUImage;
    auto CPUStart = std::chrono::high_resolution_clock::now();
    cv::GaussianBlur(inputImage, CPUImage, cv::Size(kernelSize, kernelSize), 0);
    auto CPUEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> CPUElapsed = CPUEnd - CPUStart;
    std::cout << "Execution time of CPU blur, with sigma = " << sigma << " is: " << CPUElapsed.count() << " ms\n";

    cv::imshow("CPU image", CPUImage);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <Image_Path>\n";
        return 1;
    }

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Image not found or unable to open\n";
        return 1;
    }

    compareBlur(image, 10.0);
    cv::waitKey(0);
    
    return 0;
}
