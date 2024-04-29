#include "image_processor.hpp"
#include <iostream>

int main(int argc, char** argv) {
    std::cout << "Starting the image processing program.\n";

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path>\n";
        return 1; 
    }

    ImageProcessor processor;

    // processor.readFromFile(argv[1]);
    // processor.displayImage("input image");
    processor.processAllImagesInDirectory("../assets");

    return 0;
}
