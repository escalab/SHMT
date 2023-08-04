#include <math.h>
#include <string>
#include <stdio.h>
#include "kernels_cpu.h"
#include "kernels_gpu.h"
#include "BmpUtil.h"
#include "dct8x8_kernel2.cuh"
#include "dct8x8_kernel_quantization.cuh"

#define BENCHMARK_SIZE 10
#define BLOCK_SIZE 8
#define BLOCK_SIZE2 64
#define BLOCK_SIZE_LOG2 3

//float C_a = 1.387039845322148f; //!< a = (2^0.5) * cos(    pi / 16);  Used in forward and inverse DCT.
//float C_b = 1.306562964876377f; //!< b = (2^0.5) * cos(    pi /  8);  Used in forward and inverse DCT.
//float C_c = 1.175875602419359f; //!< c = (2^0.5) * cos(3 * pi / 16);  Used in forward and inverse DCT.
//float C_d = 0.785694958387102f; //!< d = (2^0.5) * cos(5 * pi / 16);  Used in forward and inverse DCT.
//float C_e = 0.541196100146197f; //!< e = (2^0.5) * cos(3 * pi /  8);  Used in forward and inverse DCT.
//float C_f = 0.275899379282943f; //!< f = (2^0.5) * cos(7 * pi / 16);  Used in forward and inverse DCT.
//float C_norm = 0.3535533905932737f; // 1 / (8^0.5)

/**
*  JPEG quality=0_of_12 quantization matrix
*/
float Q_array[BLOCK_SIZE2] =
{
    32.f,  33.f,  51.f,  81.f,  66.f,  39.f,  34.f,  17.f,
    33.f,  36.f,  48.f,  47.f,  28.f,  23.f,  12.f,  12.f,
    51.f,  48.f,  47.f,  28.f,  23.f,  12.f,  12.f,  12.f,
    81.f,  47.f,  28.f,  23.f,  12.f,  12.f,  12.f,  12.f,
    66.f,  28.f,  23.f,  12.f,  12.f,  12.f,  12.f,  12.f,
    39.f,  23.f,  12.f,  12.f,  12.f,  12.f,  12.f,  12.f,
    34.f,  12.f,  12.f,  12.f,  12.f,  12.f,  12.f,  12.f,
    17.f,  12.f,  12.f,  12.f,  12.f,  12.f,  12.f,  12.f
};

void SubroutineDCTvector(float *FirstIn, int StepIn, float *FirstOut, int StepOut)
{
    float X07P = FirstIn[0*StepIn] + FirstIn[7*StepIn];
    float X16P = FirstIn[1*StepIn] + FirstIn[6*StepIn];
    float X25P = FirstIn[2*StepIn] + FirstIn[5*StepIn];
    float X34P = FirstIn[3*StepIn] + FirstIn[4*StepIn];

    float X07M = FirstIn[0*StepIn] - FirstIn[7*StepIn];
    float X61M = FirstIn[6*StepIn] - FirstIn[1*StepIn];
    float X25M = FirstIn[2*StepIn] - FirstIn[5*StepIn];
    float X43M = FirstIn[4*StepIn] - FirstIn[3*StepIn];

    float X07P34PP = X07P + X34P;
    float X07P34PM = X07P - X34P;
    float X16P25PP = X16P + X25P;
    float X16P25PM = X16P - X25P;

    FirstOut[0*StepOut] = C_norm * (X07P34PP + X16P25PP);
    FirstOut[2*StepOut] = C_norm * (C_b * X07P34PM + C_e * X16P25PM);
    FirstOut[4*StepOut] = C_norm * (X07P34PP - X16P25PP);
    FirstOut[6*StepOut] = C_norm * (C_e * X07P34PM - C_b * X16P25PM);

    FirstOut[1*StepOut] = C_norm * (C_a * X07M - C_c * X61M + C_d * X25M - C_f * X43M);
    FirstOut[3*StepOut] = C_norm * (C_c * X07M + C_f * X61M - C_a * X25M + C_d * X43M);
    FirstOut[5*StepOut] = C_norm * (C_d * X07M + C_a * X61M + C_f * X25M - C_c * X43M);
    FirstOut[7*StepOut] = C_norm * (C_f * X07M + C_d * X61M + C_c * X25M + C_a * X43M);
}

void computeDCT8x8Gold2(const float *fSrc, float *fDst, int Stride, int width, int height){
    for (int i = 0; i + BLOCK_SIZE - 1 < height; i += BLOCK_SIZE)
    {
        for (int j = 0; j + BLOCK_SIZE - 1 < width; j += BLOCK_SIZE)
        {
            //process rows
            for (int k = 0; k < BLOCK_SIZE; k++)
            {
                SubroutineDCTvector((float *)fSrc + (i+k) * Stride + j, 1, fDst + (i+k) * Stride + j, 1);
            }

            //process columns
            for (int k = 0; k < BLOCK_SIZE; k++)
            {
                SubroutineDCTvector(fDst + i * Stride + (j+k), Stride, fDst + i * Stride + (j+k), Stride);
            }
        }
    }
}

/*Already implemented in utils/BmpUtil.cpp*/
//float round_f(float num)
//{
//    float NumAbs = fabs(num);
//    int NumAbsI = (int)(NumAbs + 0.5f);
//    float sign = num > 0 ? 1.0f : -1.0f;
//    return sign * NumAbsI;
//}

void quantizeGoldFloat(float *fSrcDst, int Stride, int width, int height)
{

    //perform block wise in-place quantization using Q_array
    //Q_array(A) = round(A ./ Q_array) .* Q_array;
    for (int i=0; i<height; i++)
    {
        for (int j=0; j<width; j++)
        {
            int qx = j % BLOCK_SIZE;
            int qy = i % BLOCK_SIZE;
            float quantized = round_f(fSrcDst[i*Stride+j] / Q_array[(qy<<BLOCK_SIZE_LOG2)+qx]);
            fSrcDst[i*Stride+j] = quantized * Q_array[(qy<<BLOCK_SIZE_LOG2)+qx];
        }
    }
}

void SubroutineIDCTvector(float *FirstIn, int StepIn, float *FirstOut, int StepOut)
{   
    float Y04P   = FirstIn[0*StepIn] + FirstIn[4*StepIn];
    float Y2b6eP = C_b * FirstIn[2*StepIn] + C_e * FirstIn[6*StepIn];
    
    float Y04P2b6ePP = Y04P + Y2b6eP;
    float Y04P2b6ePM = Y04P - Y2b6eP;
    float Y7f1aP3c5dPP = C_f * FirstIn[7*StepIn] + C_a * FirstIn[1*StepIn] + C_c * FirstIn[3*StepIn] + C_d * FirstIn[5*StepIn];
    float Y7a1fM3d5cMP = C_a * FirstIn[7*StepIn] - C_f * FirstIn[1*StepIn] + C_d * FirstIn[3*StepIn] - C_c * FirstIn[5*StepIn];
    
    float Y04M   = FirstIn[0*StepIn] - FirstIn[4*StepIn];
    float Y2e6bM = C_e * FirstIn[2*StepIn] - C_b * FirstIn[6*StepIn];
    
    float Y04M2e6bMP = Y04M + Y2e6bM;
    float Y04M2e6bMM = Y04M - Y2e6bM;
    float Y1c7dM3f5aPM = C_c * FirstIn[1*StepIn] - C_d * FirstIn[7*StepIn] - C_f * FirstIn[3*StepIn] - C_a * FirstIn[5*StepIn];
    float Y1d7cP3a5fMM = C_d * FirstIn[1*StepIn] + C_c * FirstIn[7*StepIn] - C_a * FirstIn[3*StepIn] + C_f * FirstIn[5*StepIn];
    
    FirstOut[0*StepOut] = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
    FirstOut[7*StepOut] = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
    FirstOut[4*StepOut] = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
    FirstOut[3*StepOut] = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);
    
    FirstOut[1*StepOut] = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
    FirstOut[5*StepOut] = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
    FirstOut[2*StepOut] = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
    FirstOut[6*StepOut] = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);
}

void computeIDCT8x8Gold2(const float *fSrc, float *fDst, int Stride, int width, int height)
{
    for (int i = 0; i + BLOCK_SIZE - 1 < height; i += BLOCK_SIZE)
    {
        for (int j = 0; j + BLOCK_SIZE - 1 < width; j += BLOCK_SIZE)
        {
            //process rows
            for (int k = 0; k < BLOCK_SIZE; k++)
            {
                SubroutineIDCTvector((float *)fSrc + (i+k) * Stride + j, 1, fDst + (i+k) * Stride + j, 1);
            }

            //process columns
            for (int k = 0; k < BLOCK_SIZE; k++)
            {
                SubroutineIDCTvector(fDst + i * Stride + (j+k), Stride, fDst + i * Stride + (j+k), Stride);
            }
        }
    }
}


/*
    CPU dct8x8. This implementation requires both dimensions of input/output 
    arrays to be multiply of 8. Incorrect otherwise.
    Reference: samples/3_Imaging/dct8x8/dct8x8.cu: Gold2
*/
void CpuKernel::dct8x8_2d(Params params, float* input, float* output){
    int width  = params.get_kernel_size();
    int height = params.get_kernel_size();
    int StrideF = width;
    
    float* ImgF2 = (float*) malloc(width * height * sizeof(float));  
    
    assert(width%8==0 && height%8==0);

    for(int i = 0 ; i < BENCHMARK_SIZE ; i++){
        computeDCT8x8Gold2(input, ImgF2, StrideF, width, height);
    }
    quantizeGoldFloat(ImgF2, StrideF, width, height);
    computeIDCT8x8Gold2(ImgF2, output, StrideF, width, height);

//    ROI Size;
//    Size.width  = params.get_kernel_size();
//    Size.height = params.get_kernel_size();
//    AddFloatPlane(128.0f, output, StrideF, Size);
}

/*
    GPU dct8x8
    Reference: samples/3_Imaging/dct8x8/dct8x8.cu: CUDA2
*/
void GpuKernel::dct8x8_2d(KernelParams& kernel_params, void** in_img, void** out_img){
    /* integration code */
    float* ImgF1   = reinterpret_cast<float*>(*in_img);
    float* out_tmp = reinterpret_cast<float*>(*out_img);
    // a hard-coded params that used by this kernel.
    //int ImgStride;

    //allocate device memory
    float *src, *dst;
    ROI Size;
    Size.width  = kernel_params.params.get_kernel_size();
    Size.height = kernel_params.params.get_kernel_size();
    /* integration code */
    int StrideF = (((int)ceil((Size.width*sizeof(float))/16.0f))*16) / sizeof(float);
 //   byte *ImgDst = MallocPlaneByte(Size.width, Size.height, &ImgStride);
    size_t DeviceStride;
    checkCudaErrors(cudaMallocPitch((void **)&src, &DeviceStride, Size.width * sizeof(float), Size.height));
    checkCudaErrors(cudaMallocPitch((void **)&dst, &DeviceStride, Size.width * sizeof(float), Size.height));
    DeviceStride /= sizeof(float);

    //copy from host memory to device
    checkCudaErrors(cudaMemcpy2D(src, DeviceStride * sizeof(float),
                                 ImgF1, StrideF * sizeof(float),
                                 Size.width * sizeof(float), Size.height,
                                 cudaMemcpyHostToDevice));

    dim3 GridFullWarps(Size.width / KER2_BLOCK_WIDTH, Size.height / KER2_BLOCK_HEIGHT, 1);
    dim3 ThreadsFullWarps(8, KER2_BLOCK_WIDTH/8, KER2_BLOCK_HEIGHT/8);

    //perform block-wise DCT processing and benchmarking
    const int numIterations = 100;

    for (int i = -1; i < numIterations; i++)
    {
        if (i == 0)
        {
            checkCudaErrors(cudaDeviceSynchronize());
        }
 
        CUDAkernel2DCT<<<GridFullWarps, ThreadsFullWarps>>>(dst, src, (int)DeviceStride);
        getLastCudaError("Kernel execution failed");
    }
 
    checkCudaErrors(cudaDeviceSynchronize());

    //setup execution parameters for quantization
    dim3 ThreadsSmallBlocks(BLOCK_SIZE, BLOCK_SIZE);
    dim3 GridSmallBlocks(Size.width / BLOCK_SIZE, Size.height / BLOCK_SIZE);

    // execute Quantization kernel
    CUDAkernelQuantizationFloat<<< GridSmallBlocks, ThreadsSmallBlocks >>>(dst, (int) DeviceStride);
    getLastCudaError("Kernel execution failed");
 
    //perform block-wise IDCT processing
    CUDAkernel2IDCT<<<GridFullWarps, ThreadsFullWarps >>>(src, dst, (int)DeviceStride);
    checkCudaErrors(cudaDeviceSynchronize());
    getLastCudaError("Kernel execution failed");
    
    //copy quantized image block to host
    checkCudaErrors(cudaMemcpy2D(out_tmp, StrideF *sizeof(float),
                                 src, DeviceStride *sizeof(float),
                                 Size.width *sizeof(float), Size.height,
                                 cudaMemcpyDeviceToHost));
 
    //convert image back to byte representation
//    AddFloatPlane(128.0f, out_tmp, StrideF, Size);
//    CopyFloat2Byte(out_tmp, StrideF, ImgDst, ImgStride, Size);

    //clean up memory
    checkCudaErrors(cudaFree(dst));
    checkCudaErrors(cudaFree(src));
}


