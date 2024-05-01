#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

extern void convertToGrayscale(unsigned char* inputImage, unsigned char* outputImage, int height, int width, size_t inputStep, size_t outputStep);

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Image cannot be loaded." << std::endl;
        return 1;
    }

    // cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cv::imshow("Input Image", image);

    cv::Mat outputImage(image.rows, image.cols, CV_8UC1);

    // convertToGrayscale(image.data, outputImage.data, image.rows, image.cols, image.step, outputImage.step);
    


    cv::imshow("Grayscale Image", outputImage);
    cv::waitKey(0);

    return 0;
}