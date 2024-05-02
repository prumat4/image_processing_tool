#include "CudaImageProcessor.cuh"

#define cudaCheckError() { \
    cudaError_t e = cudaGetLastError(); \
    if(e != cudaSuccess) { \
        std::cout << "CUDA error " << cudaGetErrorString(e) << " at line " << __LINE__ << std::endl; \
    } \
}

constexpr double sigma = 3.0;

CudaImageProcessor::CudaImageProcessor(cv::Mat& input) 
: inputImage(input), outputImage(input.rows, input.cols, CV_8UC3) { 
    numInputBytes = inputImage.rows * inputImage.step;
    numOutputBytes = inputImage.rows * inputImage.step;

    cudaMalloc(&d_input, numInputBytes);
    cudaMalloc(&d_output, numOutputBytes);
    cudaMemcpy(d_input, inputImage.data, numInputBytes, cudaMemcpyHostToDevice);
}
CudaImageProcessor::~CudaImageProcessor() {
    cudaFree(d_input);
    cudaFree(d_output);
}

void CudaImageProcessor::processImage(KernelFunc kernel) {
    dim3 blockSize(16, 16);
    dim3 gridSize((inputImage.cols + blockSize.x - 1) / blockSize.x,
                  (inputImage.rows + blockSize.y - 1) / blockSize.y);
    kernel<<<gridSize, blockSize>>>(d_input, d_output, inputImage.cols, inputImage.rows);
    cudaMemcpy(outputImage.data, d_output, numOutputBytes, cudaMemcpyDeviceToHost);
}

void CudaImageProcessor::greyscale() {
    processImage(colorToGreyscaleKernel);
}

void CudaImageProcessor::sepia() {
    processImage(colorToSepiaKernel);
}

void CudaImageProcessor::invert() {
    processImage(colorToInvertedKernel);
}

void CudaImageProcessor::binary() {
    processImage(colorToBinaryKernel);
}

void CudaImageProcessor::cooling() {
    processImage(colorToCoolingKernel);
}

void CudaImageProcessor::redBoost() {
    processImage(colorToRedBoostKernel);
}
    
void CudaImageProcessor::rotate() {
    processImage(rotateKernel);
}

void CudaImageProcessor::blur() {
    double* d_kernel = nullptr;
    int kernelSize = 6 * sigma + 1; 
    size_t kernelBytes = kernelSize * kernelSize * sizeof(double);

    cudaMalloc(&d_kernel, kernelBytes);
    
    dim3 blockSizeKernel(1, 1); 
    dim3 gridSizeKernel(1, 1);
    
    generateGaussianKernelDevice<<<gridSizeKernel, blockSizeKernel>>>(d_kernel, kernelSize, sigma);
    cudaCheckError();

    dim3 blockSize(16, 16);
    dim3 gridSize((inputImage.cols + blockSize.x - 1) / blockSize.x,
                  (inputImage.rows + blockSize.y - 1) / blockSize.y);
    
    gaussianBlurKernel<<<gridSize, blockSize>>>(d_input, d_output, inputImage.cols, inputImage.rows, d_kernel, kernelSize);
    cudaCheckError();

    cudaMemcpy(outputImage.data, d_output, numOutputBytes, cudaMemcpyDeviceToHost);
    cudaCheckError();

    cudaFree(d_kernel); 
}

cv::Mat CudaImageProcessor::getOutputImage() {
    return outputImage;
}

__global__ void generateGaussianKernelDevice(double* kernel, int kernelSize, double sigma) {
    double sum = 0.0;
    int center = kernelSize / 2;
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            int x = i - center;
            int y = j - center;
            double exponent = -(x * x + y * y) / (2 * sigma * sigma);
            kernel[i * kernelSize + j] = exp(exponent) / (2 * M_PI * sigma * sigma);
            sum += kernel[i * kernelSize + j];
        }
    }

    for (int i = 0; i < kernelSize * kernelSize; ++i) {
        kernel[i] /= sum;
    }
}

__global__ void colorToBinaryKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width * 3 + x * 3;
    unsigned char b = input[idx];
    unsigned char g = input[idx + 1];
    unsigned char r = input[idx + 2];
    unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    unsigned char binary = gray > 100 ? 255 : 0;

    output[idx] = binary;
    output[idx + 1] = binary;
    output[idx + 2] = binary;
}

__global__ void colorToCoolingKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width * 3 + x * 3;
    unsigned char b = input[idx];
    unsigned char g = input[idx + 1];
    unsigned char r = input[idx + 2];

    float coolB = 1.2f * b;
    coolB = coolB > 255 ? 255 : coolB;

    output[idx] = static_cast<unsigned char>(coolB);
    output[idx + 1] = g;
    output[idx + 2] = r;
}

__global__ void colorToInvertedKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width * 3 + x * 3;
    unsigned char b = input[idx];
    unsigned char g = input[idx + 1];
    unsigned char r = input[idx + 2];

    output[idx] = 255 - b;
    output[idx + 1] = 255 - g;
    output[idx + 2] = 255 - r;
}

__global__ void colorToSepiaKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width * 3 + x * 3;
    unsigned char b = input[idx];
    unsigned char g = input[idx + 1];
    unsigned char r = input[idx + 2];

    float sepiaR = 0.393f * r + 0.769f * g + 0.189f * b;
    float sepiaG = 0.349f * r + 0.686f * g + 0.168f * b;
    float sepiaB = 0.272f * r + 0.534f * g + 0.131f * b;

    sepiaR = sepiaR > 255 ? 255 : sepiaR;
    sepiaG = sepiaG > 255 ? 255 : sepiaG;
    sepiaB = sepiaB > 255 ? 255 : sepiaB;

    output[idx] = static_cast<unsigned char>(sepiaB);
    output[idx + 1] = static_cast<unsigned char>(sepiaG);
    output[idx + 2] = static_cast<unsigned char>(sepiaR);
}

__global__ void colorToRedBoostKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width * 3 + x * 3;
    unsigned char b = input[idx];
    unsigned char g = input[idx + 1];
    unsigned char r = input[idx + 2];

    float boostR = 1.2f * r;
    boostR = boostR > 255 ? 255 : boostR;

    output[idx] = b;
    output[idx + 1] = g;
    output[idx + 2] = static_cast<unsigned char>(boostR);
}

__global__ void colorToGreyscaleKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width * 3 + x * 3;
    unsigned char b = input[idx];
    unsigned char g = input[idx + 1];
    unsigned char r = input[idx + 2];
    unsigned char gray = static_cast<unsigned char>(0.114f * b + 0.587f * g + 0.299f * r);
    output[idx] = gray;    
    output[idx + 1] = gray;
    output[idx + 2] = gray;
}

__global__ void rotateKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width * 3 + x * 3;
    int newX = width - 1 - x;
    int newY = height - 1 - y;
    int newIdx = newY * width * 3 + newX * 3;
    
    output[newIdx]     = input[idx];
    output[newIdx + 1] = input[idx + 1];
    output[newIdx + 2] = input[idx + 2];
}

__global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output, int width, int height, double* kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int halfKernel = kernelSize / 2;
    double redSum = 0.0, greenSum = 0.0, blueSum = 0.0;

    for (int i = -halfKernel; i <= halfKernel; i++) {
        for (int j = -halfKernel; j <= halfKernel; j++) {
            int nx = x + i;
            int ny = y + j;

            nx = max(0, min(nx, width - 1));
            ny = max(0, min(ny, height - 1));

            int imgIndex = (ny * width + nx) * 3;
            int kernIndex = (i + halfKernel) * kernelSize + (j + halfKernel);
            double kernelVal = kernel[kernIndex];

            blueSum  += input[imgIndex]     * kernelVal;
            greenSum += input[imgIndex + 1] * kernelVal;
            redSum   += input[imgIndex + 2] * kernelVal;
        }
    }

    int outputIndex = (y * width + x) * 3;
    output[outputIndex] = static_cast<unsigned char>(blueSum);   
    output[outputIndex + 1] = static_cast<unsigned char>(greenSum); 
    output[outputIndex + 2] = static_cast<unsigned char>(redSum); 
}