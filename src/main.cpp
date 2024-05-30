#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "CudaImageProcessor.cuh"

namespace fs = std::filesystem;

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

    CudaImageProcessor processor(image);
    cv::imshow("Input image", image);

    processor.timeExecution("Canny edge detection", processor, &CudaImageProcessor::cannyEdgeDetection, 1.1);

    cv::Mat outputImage = processor.getOutputImage();
    // cv::imwrite("../assets/blur/1600x900_blur.jpg", outputImage);
    cv::imshow("Output image", outputImage);
    cv::waitKey(0);
    
    return 0;
}
