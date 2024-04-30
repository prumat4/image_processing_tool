#include "image_processor.hpp"
#include <iostream>

extern void addWithCuda(int *c, const int *a, const int *b, int size);

int main() {
    const int arraySize = 5;
    int a[arraySize] = {1, 2, 3, 4, 5};
    int b[arraySize] = {10, 20, 30, 40, 50};
    int c[arraySize] = {0};

    addWithCuda(c, a, b, arraySize);

    for (int i = 0; i < arraySize; i++) {
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    }

    return 0;
}

// int main(int argc, char** argv) {
//     std::cout << "Starting the image processing program.\n";

//     if (argc < 2) {
//         std::cerr << "Usage: " << argv[0] << " <path>\n";
//         return 1; 
//     }

//     ImageProcessor processor;

//     processor.readFromFile(argv[1]);
//     processor.displayImage("input image");
//     // processor.processAllImagesInDirectory("../assets");

//     Wrapper::wrapper();

//     return 0;
// }
