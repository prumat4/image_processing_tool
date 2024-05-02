#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "CudaImageProcessor.cuh"

namespace fs = std::filesystem;

int main() {
    std::string inputDirectory = "../assets/";
    std::string outputDirectory = "../assets/edgeDetector";
    
    fs::create_directories(outputDirectory);

    for (const auto& entry : fs::directory_iterator(inputDirectory)) {
        if (entry.is_regular_file() && (entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg")) {
            cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
            if (image.empty()) {
                std::cerr << "Error: Unable to open " << entry.path() << std::endl;
                continue;
            }

            CudaImageProcessor processor(image);
            processor.timeExecution("Gaussian blur", &CudaImageProcessor::cannyEdgeDetection);
            cv::Mat outputImage = processor.getOutputImage();

            std::string outputFilePath = outputDirectory + "/" + entry.path().stem().string() + "_edge.jpg";
            cv::imwrite(outputFilePath, outputImage);
        }
    }
    
    return 0;
}

#include <iostream>
#include "CudaImageProcessor.cuh"

// int main(int argc, char** argv) {
//     if (argc < 2) {
//         std::cerr << "Usage: " << argv[0] << " <Image_Path>\n";
//         return 1;
//     }

//     cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
//     if (image.empty()) {
//         std::cerr << "Error: Image not found or unable to open\n";
//         return 1;
//     }

//     CudaImageProcessor processor(image);
//     cv::imshow("Input image", image);
//     // processor.timeExecution("Gaussian blur", &CudaImageProcessor::blur);
//     // processor.timeExecution("Grayscale Conversion", &CudaImageProcessor::greyscale);
//     // processor.timeExecution("Rotate Conversion", &CudaImageProcessor::rotate);

//     // processor.timeExecution("Grayscale Conversion", &CudaImageProcessor::blur);

//     processor.timeExecution("Canny Edge Detection", &CudaImageProcessor::cannyEdgeDetection);

//     cv::Mat outputImage = processor.getOutputImage();
//     // cv::imwrite("../assets/blur/1600x900_blur.jpg", outputImage);
//     cv::imshow("Output image", outputImage);
//     cv::waitKey(0);
    
//     return 0;
// }
