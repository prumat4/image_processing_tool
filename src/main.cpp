#include <opencv4/opencv2/opencv.hpp>

#include <iostream>

int main(int argc, char** argv) {
    std::cout << "Starting the image processing program.\n";

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <Image_Path>\n";
        return 1; 
    }

    std::cout << "Attempting to open image at: " << argv[1] << std::endl;

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Image not found or unable to open\n";
        return 1;
    }

    std::cout << "Image loaded successfully.\n";
    
    cv::Mat new_image(image.rows, image.cols, image.type());

    for(int r = 0; r < new_image.rows; r++) {
        for(int c = 0; c < new_image.cols; c++) {
        cv::Vec3b old_color = image.at<cv::Vec3b>(r, c);
            unsigned char gray = static_cast<unsigned char>(
                (float)old_color[0] * 0.114f +
                (float)old_color[1] * 0.587f +
                (float)old_color[2] * 0.299f 
            );

            new_image.at<cv::Vec3b>(r, c) = cv::Vec3b(gray, gray, gray);
        }
    }

    if (!cv::imwrite("../assets/new_image.jpg", new_image)) { 
        std::cerr << "Failed to write the new image." << std::endl;
        return 1;
    }
    
    cv::imshow("greyish Image", new_image);

    cv::waitKey(0);

    return 0;
}
