#include <iostream>
#include <cuda_runtime.h>

__global__ void grayscaleKernel(unsigned char* inputImage, unsigned char* outputImage, int height, int width, size_t inputStep, size_t outputStep) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int inputIdx = y * inputStep + 3 * x;
    int outputIdx = y * outputStep + x;

    inputIdx = min(inputIdx, static_cast<int>((height * inputStep) - 3)); 
    outputIdx = min(outputIdx, static_cast<int>((height * outputStep) - 1));

    unsigned char b = inputImage[inputIdx];
    unsigned char g = inputImage[inputIdx + 1];
    unsigned char r = inputImage[inputIdx + 2];

    float grayscale = 0.114f * b + 0.587f * g + 0.299f * r;
    outputImage[outputIdx] = static_cast<unsigned char>(grayscale);
}

void convertToGrayscale(unsigned char* inputImage, unsigned char* outputImage, int height, int width, size_t inputStep, size_t outputStep) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    grayscaleKernel<<<gridSize, blockSize>>>(inputImage, outputImage, height, width, inputStep, outputStep);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}