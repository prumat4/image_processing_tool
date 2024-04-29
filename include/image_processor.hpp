#pragma once

#include <opencv4/opencv2/opencv.hpp>
#include <filesystem>

class ImageProcessor {
public:
    explicit ImageProcessor(const cv::Mat& _image);
    ImageProcessor() = default;

    // no need?
    void readFromFile(const std::string& _filePath);
    void displayImage(const std::string& _name) const;

    // in future, this should take some id of a function to execute
    void processAllImagesInDirectory(const std::string& directoryPath);
    static void makeGreyscaleAndSave(const std::string& inputPath);

private: 
    static bool isImageFile(const std::string& filename);

private:
    // no need?
    cv::Mat image;
    
    std::string filePath;
};