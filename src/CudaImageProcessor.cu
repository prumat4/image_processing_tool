#include "CudaImageProcessor.cuh"

CudaImageProcessor::CudaImageProcessor(cv::Mat& input) 
: inputImage(input), outputImage(input.rows, input.cols, CV_8UC1) {
    numInputBytes = inputImage.rows * inputImage.step;
    numOutputBytes = inputImage.rows * inputImage.cols;

    cudaMalloc(&d_input, numInputBytes);
    cudaMalloc(&d_output, numOutputBytes);
    cudaMemcpy(d_input, inputImage.data, numInputBytes, cudaMemcpyHostToDevice);
}

CudaImageProcessor::~CudaImageProcessor() {
    cudaFree(d_input);
    cudaFree(d_output);
}

void CudaImageProcessor::convertToGreyscale() {
    dim3 blockSize(16, 16);
    dim3 gridSize((inputImage.cols + blockSize.x - 1) / blockSize.x,
                  (inputImage.rows + blockSize.y - 1) / blockSize.y);
    colorToGrayscaleKernel<<<gridSize, blockSize>>>(d_input, d_output, inputImage.cols, inputImage.rows);
    cudaMemcpy(outputImage.data, d_output, numOutputBytes, cudaMemcpyDeviceToHost);
}

cv::Mat CudaImageProcessor::getOutputImage() {
    return outputImage;
}

__global__ void colorToGrayscaleKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width * 3 + x * 3;
        unsigned char b = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char r = input[idx + 2];
        output[y * width + x] = static_cast<unsigned char>(0.114f * b + 0.587f * g + 0.299f * r);
    }
}
