#include <iostream>
#include "CudaImageProcessor.cuh"

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
    // processor.timeExecution("Rotate Conversion", &CudaImageProcessor::rotate);
    // processor.timeExecution("Grayscale Conversion", &CudaImageProcessor::convertToGreyscale);
    processor.timeExecution("Gaussian blur", &CudaImageProcessor::blur);

    cv::Mat outputImage = processor.getOutputImage();

    cv::imshow("Output image", outputImage);
    cv::waitKey(0);
    
    return 0;
}
