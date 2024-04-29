#include "image_processor.hpp"

ImageProcessor::ImageProcessor(const cv::Mat& _image) : image(_image.clone()) {}

void ImageProcessor::readFromFile(const std::string& _filePath) {
    std::cout << "Attempting to open image at: " << _filePath << std::endl;
    filePath = _filePath;
    image = cv::imread(filePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to load the image from the file path provided." << std::endl;
    }
}

void ImageProcessor::displayImage(const std::string& _name) const {
    if (image.empty()) {
        std::cerr << "Image not loaded." << std::endl;
        return;
    }
    cv::imshow(_name, image);
    cv::waitKey(0);
}

void ImageProcessor::processAllImagesInDirectory(const std::string& directoryPath) {
    for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
        if (entry.is_regular_file() && isImageFile(entry.path().filename().string())) {
            makeGreyscaleAndSave(entry.path().string());
        }
    }
}

void ImageProcessor::makeGreyscaleAndSave(const std::string& inputPath) {
    cv::Mat image = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to load the image from: " << inputPath << std::endl;
        return;
    }

    cv::Mat newImage;
    cv::cvtColor(image, newImage, cv::COLOR_BGR2GRAY);

    std::filesystem::path outputPath = std::filesystem::path(inputPath).parent_path() / "greyscale";
    std::filesystem::create_directories(outputPath);

    std::string outputFileName = std::filesystem::path(inputPath).stem().string() + "_grayscale.jpg";
    std::string fullOutputPath = outputPath / outputFileName;

    if (!cv::imwrite(fullOutputPath, newImage)) {
        std::cerr << "Failed to save the grayscale image to: " << fullOutputPath << std::endl;
    } else {
        std::cout << "Grayscale image saved to: " << fullOutputPath << std::endl;
    }
}

bool ImageProcessor::isImageFile(const std::string& filename) {
    std::vector<std::string> extensions = { ".png", ".jpg", ".jpeg", ".bmp" };
    std::filesystem::path filePath(filename);
    return std::find(extensions.begin(), extensions.end(), filePath.extension().string()) != extensions.end();
}