#include <assert.h>
#include <iostream>
#include "cuda_utils.h"
#include "kernels_fft.cuh"

__global__ void test_kernel(){
  printf("Hello World!\n");
}

////////////////////////////////////////////////////////////////////////////////
/// Position convolution kernel center at (0, 0) in the image
////////////////////////////////////////////////////////////////////////////////
extern "C" void padKernel(
    float *d_Dst,
    float *d_Src,
    int fftH,
    int fftW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
)
{
    assert(d_Src != d_Dst);
    dim3 threads(32, 8);
    dim3 grid(iDivUp(kernelW, threads.x), iDivUp(kernelH, threads.y));
 
    SET_FLOAT_BASE;
#if (USE_TEXTURE)
    cudaTextureObject_t texFloat;
    cudaResourceDesc    texRes;
    memset(&texRes,0,sizeof(cudaResourceDesc));
 
    texRes.resType            = cudaResourceTypeLinear;
    texRes.res.linear.devPtr    = d_Src;
    texRes.res.linear.sizeInBytes = sizeof(float)*kernelH*kernelW;
    texRes.res.linear.desc = cudaCreateChannelDesc<float>();

    cudaTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = false;
    texDescr.filterMode       = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;
  
    cudaCreateTextureObject(&texFloat, &texRes, &texDescr, NULL);
#endif
  
    padKernel_kernel<<<grid, threads>>>(
        d_Dst,
        d_Src,
        fftH,
        fftW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
#if (USE_TEXTURE)
        , texFloat
#endif
    );
//    getLastCudaError("padKernel_kernel<<<>>> execution failed\n");
 
#if (USE_TEXTURE)
    cudaDestroyTextureObject(texFloat);
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Prepare data for "pad to border" addressing mode
////////////////////////////////////////////////////////////////////////////////
extern "C" void padDataClampToBorder(
    float *d_Dst,
    float *d_Src,
    int fftH,
    int fftW,
    int dataH,
    int dataW,
    int kernelW,
    int kernelH,
    int kernelY,
    int kernelX
)
{
    assert(d_Src != d_Dst);
    dim3 threads(32, 8);
    dim3 grid(iDivUp(fftW, threads.x), iDivUp(fftH, threads.y));

#if (USE_TEXTURE)
    cudaTextureObject_t texFloat;
    cudaResourceDesc            texRes;
    memset(&texRes,0,sizeof(cudaResourceDesc));

    texRes.resType            = cudaResourceTypeLinear;
    texRes.res.linear.devPtr    = d_Src;
    texRes.res.linear.sizeInBytes = sizeof(float)*dataH*dataW;
    texRes.res.linear.desc = cudaCreateChannelDesc<float>();
 
    cudaTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(cudaTextureDesc));
 
    texDescr.normalizedCoords = false;
    texDescr.filterMode       = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;
 
    cudaCreateTextureObject(&texFloat, &texRes, &texDescr, NULL);
#endif
 
    padDataClampToBorder_kernel<<<grid, threads>>>(
        d_Dst,
        d_Src,
        fftH,
        fftW,
        dataH,
        dataW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
#if (USE_TEXTURE)
       ,texFloat
#endif
    );
//    getLastCudaError("padDataClampToBorder_kernel<<<>>> execution fai    led\n");
 
#if (USE_TEXTURE)
    cudaDestroyTextureObject(texFloat);
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
extern "C" void modulateAndNormalize(
    fComplex *d_Dst,
    fComplex *d_Src,
    int fftH,
    int fftW,
    int padding
)
{
    assert(fftW % 2 == 0);
    const int dataSize = fftH * (fftW / 2 + padding);

    modulateAndNormalize_kernel<<<iDivUp(dataSize, 256), 256>>>(
        d_Dst,
        d_Src,
        dataSize,
        1.0f / (float)(fftW *fftH)
    );
//    getLastCudaError("modulateAndNormalize() execution failed\n");
}


void fft_2d_input_conversion_wrapper(){
    return;
}

void fft_2d_kernel_wrapper(float* in_img, float* out_img) {
  std::cout << __func__ << ": calling a testing cuda kernel function..." << std::endl;
  test_kernel<<<1, 1>>>();
  return;
}
