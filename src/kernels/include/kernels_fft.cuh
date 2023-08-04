#ifndef TEST_CUH__
#define TEST_CUH__

#include <stdio.h>
#define  USE_TEXTURE 1
#define POWER_OF_TWO 1
 
 
#if(USE_TEXTURE)
#define   LOAD_FLOAT(i) tex1Dfetch<float>(texFloat, i)
#define  SET_FLOAT_BASE
#else
#define  LOAD_FLOAT(i) d_Src[i]
#define SET_FLOAT_BASE
#endif

////////////////////////////////////////////////////////////////////////////////
/// Position convolution kernel center at (0, 0) in the image
////////////////////////////////////////////////////////////////////////////////
__global__ void padKernel_kernel(
    float *d_Dst,
    float *d_Src,
    int fftH,
    int fftW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
#if (USE_TEXTURE)
    , cudaTextureObject_t texFloat
#endif
)
{
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
 
    if (y < kernelH && x < kernelW)
    {
        int ky = y - kernelY;
 
        if (ky < 0)
        {
            ky += fftH;
        }
 
        int kx = x - kernelX;
 
        if (kx < 0)
        {
            kx += fftW;
        }
 
        d_Dst[ky * fftW + kx] = LOAD_FLOAT(y * kernelW + x);
    }
}
////////////////////////////////////////////////////////////////////////////////
// Prepare data for "pad to border" addressing mode
////////////////////////////////////////////////////////////////////////////////
__global__ void padDataClampToBorder_kernel(
    float *d_Dst,
    float *d_Src,
    int fftH,
    int fftW,
    int dataH,
    int dataW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
#if (USE_TEXTURE)
    , cudaTextureObject_t texFloat
#endif
)
{
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int borderH = dataH + kernelY;
    const int borderW = dataW + kernelX;
    if (y < fftH && x < fftW)
    {
        int dy, dx;
 
        if (y < dataH)
        {
            dy = y;
        }
 
        if (x < dataW)
        {
            dx = x;
        }
 
        if (y >= dataH && y < borderH)
        {
            dy = dataH - 1;
        }
 
        if (x >= dataW && x < borderW)
        {
            dx = dataW - 1;
        }
 
        if (y >= borderH)
        {
            dy = 0;
        }
 
        if (x >= borderW)
        {
            dx = 0;
        }
        d_Dst[y * fftW + x] = LOAD_FLOAT(dy * dataW + dx);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
inline __device__ void mulAndScale(fComplex &a, const fComplex &b, const float &c)
{
    fComplex t = {c *(a.x * b.x - a.y * b.y), c *(a.y * b.x + a.x * b.y)};
    a = t;
}

__global__ void modulateAndNormalize_kernel(
    fComplex *d_Dst,
    fComplex *d_Src,
    int dataSize,
    float c
)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
 
    if (i >= dataSize)
    {
        return;
    }
 
    fComplex a = d_Src[i];
    fComplex b = d_Dst[i];
 
    mulAndScale(a, b, c);
 
    d_Dst[i] = a;
}

extern "C" void padKernel(
    float *d_PaddedKernel,
    float *d_Kernel,
    int fftH,
    int fftW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
);

void fft_2d_input_conversion_wrapper();
void fft_2d_kernel_wrapper(float* in_img, float* out_img);

#endif
