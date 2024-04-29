#include <opencv4/opencv2/opencv.hpp>

#include <iostream>

int main(int argc, char** argv) {
    std::cout << "Starting the image processing program.\n";

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <Image_Path>\n";
        return 1; 
    }

    std::cout << "Attempting to open image at: " << argv[1] << std::endl;

    cv::Mat input_image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (input_image.empty()) {
        std::cerr << "Error: Image not found or unable to open\n";
        return 1;
    }

    std::cout << "Image loaded successfully.\n";

    cv::imshow("Original Image", input_image);
    cv::waitKey(0);

    return 0;
}
