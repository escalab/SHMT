#include <math.h>
#include <string>
#include <stdio.h>
#include <cufft.h>
#include <assert.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/cudaarithm.hpp> // addWeighted()
#include <opencv2/cudafilters.hpp> // create[XXX]Filter()
#include <opencv2/cudaimgproc.hpp> // hist
#include "srad.h"
#include "BmpUtil.h"
#include "hlop_gpu.h"
#include "cuda_utils.h"
#include "srad_kernel.cu"
#include "kernels_fft.cuh"
#include "kernels_fft_wrapper.cu"
#include "dct8x8_kernel2.cuh"
#include "dct8x8_kernel_quantization.cuh"
#include "dwt.h"
#include "common.h"

#ifdef RD_WG_SIZE_0_0                                                            
        #define HOTSPOT_BLOCK_SIZE RD_WG_SIZE_0_0                                        
#elif defined(RD_WG_SIZE_0)                                                      
        #define HOTSPOT_BLOCK_SIZE RD_WG_SIZE_0                                          
#elif defined(RD_WG_SIZE)                                                        
        #define HOTSPOT_BLOCK_SIZE RD_WG_SIZE                                            
#else
        #define HOTSPOT_BLOCK_SIZE 16                                                            
#endif

/* some constants */
#define chip_height 0.016
#define chip_width 0.016
#define t_chip 0.0005
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
#define FACTOR_CHIP 0.5
#define MAX_PD 3.0e6

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
//#define MIN(a, b) ((a)<=(b) ? (a) : (b))

#define BENCHMARK_SIZE 10
#define DCT_BLOCK_SIZE 8
#define DCT_BLOCK_SIZE2 64
#define DCT_BLOCK_SIZE_LOG2 3

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
__device__ inline float cndGPU(float d)
{   
    const float       A1 = 0.31938153f;
    const float       A2 = -0.356563782f;
    const float       A3 = 1.781477937f;
    const float       A4 = -1.821255978f;
    const float       A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;
    
    float
    K = __fdividef(1.0f, (1.0f + 0.2316419f * fabsf(d)));
    
    float 
    cnd = RSQRT2PI * __expf(- 0.5f * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
    
    if (d > 0)
        cnd = 1.0f - cnd;
     
    return cnd;
}

__device__ inline void BlackScholesBodyGPU(
    float &CallResult,
    float &PutResult,
    float S, //Stock price
    float X, //Option strike
    float T, //Option years
    float R, //Riskless rate
    float V  //Volatility rate
)
{
    float sqrtT, expRT;
    float d1, d2, CNDD1, CNDD2;
 
    sqrtT = __fdividef(1.0F, rsqrtf(T));
    d1 = __fdividef(__logf(S / X) + (R + 0.5f * V * V) * T, V * sqrtT);
    d2 = d1 - V * sqrtT;
 
    CNDD1 = cndGPU(d1);
    CNDD2 = cndGPU(d2);
 
    //Calculate Call and Put simultaneously
    expRT = __expf(- R * T);
    CallResult = S * CNDD1 - X * expRT * CNDD2;
    PutResult  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}

__launch_bounds__(128)
__global__ void BlackScholesGPU(
    float2 * __restrict d_CallResult,
    float2 * __restrict d_PutResult,
    float2 * __restrict d_StockPrice,
    float2 * __restrict d_OptionStrike,
    float2 * __restrict d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
)
{
    ////Thread index
    //const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    ////Total number of threads in execution grid
    //const int THREAD_N = blockDim.x * gridDim.x;
 
    const int opt = blockDim.x * blockIdx.x + threadIdx.x;
 
     // Calculating 2 options per thread to increase ILP (instruction level para    llelism)
    if (opt < (optN / 2))
    {
        float callResult1, callResult2;
        float putResult1, putResult2;
        BlackScholesBodyGPU(
            callResult1,
            putResult1,
            d_StockPrice[opt].x,
            d_OptionStrike[opt].x,
            d_OptionYears[opt].x,
            Riskfree,
            Volatility
        );
        BlackScholesBodyGPU(
            callResult2,
            putResult2,
            d_StockPrice[opt].y,
            d_OptionStrike[opt].y,
            d_OptionYears[opt].y,
            Riskfree,
            Volatility
        );
        d_CallResult[opt] = make_float2(callResult1, callResult2);
        d_PutResult[opt] = make_float2(putResult1, putResult2);
     }
}

/*
    GPU blackscholes
    Reference: samples/4_Finance/BlackScholes
*/
void HLOPGpu::blackscholes_2d(KernelParams& kernel_params, void** in_img, void** out_img){
    int OPT_N = kernel_params.params.get_kernel_size() * kernel_params.params.get_kernel_size();
    const int OPT_SZ = OPT_N * sizeof(float);

    const float      RISKFREE = 0.02f;
    const float    VOLATILITY = 0.30f;
    
    float
    *h_CallResultGPU = (float*)*out_img,
    *h_PutResultGPU = &((float*)*out_img)[OPT_N],
    *h_StockPrice = (float*)*in_img,
    *h_OptionStrike = &((float*)*in_img)[OPT_N],
    *h_OptionYears = &((float*)*in_img)[2 * OPT_N];
    
    float
    *d_CallResult,
    *d_PutResult,
    *d_StockPrice,
    *d_OptionStrike,
    *d_OptionYears;

    //printf("...allocating GPU memory for options.\n");
    checkCudaErrors(cudaMalloc((void **)&d_CallResult,   OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_PutResult,    OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_StockPrice,   OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_OptionStrike, OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_OptionYears,  OPT_SZ));

    checkCudaErrors(cudaMemcpy(d_StockPrice,  h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionStrike, h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionYears,  h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaDeviceSynchronize());
 
    for (int i = 0; i < 1/*NUM_ITERATIONS*/; i++)
    {
        BlackScholesGPU<<<DIV_UP((OPT_N/2), 128), 128/*480, 128*/>>>(
            (float2 *)d_CallResult,
            (float2 *)d_PutResult,
            (float2 *)d_StockPrice,
            (float2 *)d_OptionStrike,
            (float2 *)d_OptionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N
        );
//        getLastCudaError("BlackScholesGPU() execution failed\n");
    }
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_PutResultGPU,  d_PutResult,  OPT_SZ, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_OptionYears));
    checkCudaErrors(cudaFree(d_OptionStrike));
    checkCudaErrors(cudaFree(d_StockPrice));
    checkCudaErrors(cudaFree(d_PutResult));
    checkCudaErrors(cudaFree(d_CallResult));
}

/*
    GPU dct8x8
    Reference: samples/3_Imaging/dct8x8/dct8x8.cu: CUDA2
*/
void HLOPGpu::dct8x8_2d(KernelParams& kernel_params, void** in_img, void** out_img){
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
    dim3 ThreadsSmallBlocks(DCT_BLOCK_SIZE, DCT_BLOCK_SIZE);
    dim3 GridSmallBlocks(Size.width / DCT_BLOCK_SIZE, Size.height / DCT_BLOCK_SIZE);

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

void HLOPGpu::fft_2d_input_conversion(){
    this->input_array_type.device_fp  = this->input_array_type.host_fp;
}

void HLOPGpu::fft_2d_output_conversion(){
    Mat result;
    const int kernelH = 7;
    const int kernelW = 6;
    const int   dataH = kernel_params.params.get_kernel_size();
    const int   dataW = kernel_params.params.get_kernel_size();
    const int    fftH = snapTransformSize(dataH + kernelH - 1); 
    const int    fftW = snapTransformSize(dataW + kernelW - 1); 

    assert(this->output_array_type.device_fp != NULL);

    array2mat(result, this->output_array_type.device_fp, fftH, fftW);
    Mat cropped = result(Range(0, dataH), Range(0, dataW)); 
    mat2array(cropped, this->output_array_type.host_fp);
    free(this->output_array_type.device_fp);
}    

/*
    GPU convolveFFT2D, this kernel used a fixed 7x6 convolving kernel.
    Reference: samples/3_Imaging/convolutionFFT2D/convolutionFFT2D.cu
*/
void HLOPGpu::fft_2d(KernelParams& kernel_params, void** in_array, void** out_array){
    float* h_Data      = reinterpret_cast<float*>(*in_array);
    float* h_ResultGPU = reinterpret_cast<float*>(*out_array);
    
    float* h_Kernel;
    
    float* d_Data ;
    float* d_PaddedData;
    float* d_Kernel;
    float* d_PaddedKernel;

    fComplex* d_DataSpectrum;
    fComplex* d_KernelSpectrum;

    cufftHandle fftPlanFwd, fftPlanInv;

    const int kernelH = 7;
    const int kernelW = 6;
    const int kernelY = 3;
    const int kernelX = 4;
    const int   dataH = kernel_params.params.get_kernel_size();
    const int   dataW = kernel_params.params.get_kernel_size();
    const int    fftH = snapTransformSize(dataH + kernelH - 1); 
    const int    fftW = snapTransformSize(dataW + kernelW - 1); 

    //printf("...allocating memory\n");
    float fft_2d_kernel_array[7*6] = {
        13, 12, 13,  0,  1,  1,
        0,  7,  8,  2,  8,  0,
        5,  9,  1, 11, 11,  3,
        14, 14,  8, 11,  0,  3,
        6,  8, 14, 13,  0, 10,
        10, 11, 14,  1,  2,  0,
        5, 15,  7,  5,  1,  7
    };
    h_Kernel = fft_2d_kernel_array;

    checkCudaErrors(cudaMalloc((void **)&d_Data, dataH * dataW * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));
    
    checkCudaErrors(cudaMalloc((void **)&d_PaddedData,   fftH * fftW * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&d_DataSpectrum,   fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    checkCudaErrors(cudaMalloc((void **)&d_KernelSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    checkCudaErrors(cudaMemset(d_KernelSpectrum, 0, fftH * (fftW / 2 + 1) * sizeof(fComplex)));

    //printf("...creating R2C & C2R FFT plans for %i x %i\n", fftH, fftW);
    checkCudaErrors(cufftPlan2d(&fftPlanFwd, fftH, fftW, CUFFT_R2C));
    checkCudaErrors(cufftPlan2d(&fftPlanInv, fftH, fftW, CUFFT_C2R));

    //printf("...uploading to gpu and padding convolution kernel and input data\n");
    checkCudaErrors(cudaMemcpy(d_Kernel, 
                               h_Kernel, 
                               kernelH * kernelW * sizeof(float), 
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Data,   
                               h_Data,   
                               dataH   * dataW *   sizeof(float), 
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)));
    checkCudaErrors(cudaMemset(d_PaddedData,   0, fftH * fftW * sizeof(float)));

    padKernel(
        d_PaddedKernel,
        d_Kernel,
        fftH,
        fftW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
    );

    padDataClampToBorder(
        d_PaddedData,
        d_Data,
        fftH,
        fftW,
        dataH,
        dataW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
    );

    //printf("...transforming convolution kernel\n");
    checkCudaErrors(cufftExecR2C(fftPlanFwd, 
                 (cufftReal *)d_PaddedKernel, 
                 (cufftComplex *)d_KernelSpectrum));
    
    //printf("...running GPU FFT convolution\n");
    checkCudaErrors(cudaDeviceSynchronize());
    
    checkCudaErrors(cufftExecR2C(fftPlanFwd, 
                                 (cufftReal *)d_PaddedData, 
                                 (cufftComplex *)d_DataSpectrum));
    modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftH, fftW, 1);
    checkCudaErrors(cufftExecC2R(fftPlanInv, 
                                 (cufftComplex *)d_DataSpectrum, 
                                 (cufftReal *)d_PaddedData));
 
    checkCudaErrors(cudaDeviceSynchronize());
    
    float* tmp = (float *)malloc(fftH    * fftW * sizeof(float));;
    
    //printf("...reading back GPU convolution results\n");
    checkCudaErrors(cudaMemcpy(tmp, d_PaddedData, fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost)); 
    h_ResultGPU = tmp;
    *out_array = (void*)h_ResultGPU;

    checkCudaErrors(cufftDestroy(fftPlanInv));
    checkCudaErrors(cufftDestroy(fftPlanFwd));

    checkCudaErrors(cudaFree(d_DataSpectrum));
    checkCudaErrors(cudaFree(d_KernelSpectrum));
    checkCudaErrors(cudaFree(d_PaddedData));
    checkCudaErrors(cudaFree(d_PaddedKernel));
    checkCudaErrors(cudaFree(d_Data));
    checkCudaErrors(cudaFree(d_Kernel));
}
 
__global__ void calculate_temp(int iteration,  //number of iteration
                               float *power,   //power input
                               float *temp_src,    //temperature input/output
                               float *temp_dst,    //temperature input/output
                               int grid_cols,  //Col of grid
                               int grid_rows,  //Row of grid
                               int border_cols,  // border offset 
                               int border_rows,  // border offset
                               float Cap,      //Capacitance
                               float Rx,
                               float Ry,
                               float Rz,
                               float step,
                               float time_elapsed){

        __shared__ float temp_on_cuda[HOTSPOT_BLOCK_SIZE][HOTSPOT_BLOCK_SIZE];
        __shared__ float power_on_cuda[HOTSPOT_BLOCK_SIZE][HOTSPOT_BLOCK_SIZE];
        __shared__ float temp_t[HOTSPOT_BLOCK_SIZE][HOTSPOT_BLOCK_SIZE]; // saving temparary temperature result
 
    float amb_temp = 80.0;
        float step_div_Cap;
        float Rx_1,Ry_1,Rz_1;

    int bx = blockIdx.x;
        int by = blockIdx.y;
 
    int tx=threadIdx.x;
    int ty=threadIdx.y;
 
    step_div_Cap=step/Cap;

    Rx_1=1/Rx;
    Ry_1=1/Ry;
    Rz_1=1/Rz;
 
        // each block finally computes result for a small block
        // after N iterations. 
        // it is the non-overlapping small blocks that cover 
        // all the input data
 
        // calculate the small block size
    int small_block_rows = HOTSPOT_BLOCK_SIZE-iteration*2;//EXPAND_RATE
    int small_block_cols = HOTSPOT_BLOCK_SIZE-iteration*2;//EXPAND_RATE

        // calculate the boundary for the block according to 
        // the boundary of its small block
        int blkY = small_block_rows*by-border_rows;
        int blkX = small_block_cols*bx-border_cols;
        int blkYmax = blkY+HOTSPOT_BLOCK_SIZE-1;
        int blkXmax = blkX+HOTSPOT_BLOCK_SIZE-1;

        // calculate the global thread coordination
    int yidx = blkY+ty;
    int xidx = blkX+tx;
 
        // load data if it is within the valid input range
    int loadYidx=yidx, loadXidx=xidx;
        int index = grid_cols*loadYidx+loadXidx;
 
    if(IN_RANGE(loadYidx, 0, grid_rows-1) && IN_RANGE(loadXidx, 0, grid_cols-1)){
            temp_on_cuda[ty][tx] = temp_src[index];  // Load the temperature data from global memory to shared memory
            power_on_cuda[ty][tx] = power[index];// Load the power data from global memory to shared memory
    }
    __syncthreads();

        // effective range within this block that falls within 
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validYmin = (blkY < 0) ? -blkY : 0;
        int validYmax = (blkYmax > grid_rows-1) ? HOTSPOT_BLOCK_SIZE-1-(blkYmax-grid_rows+1) : HOTSPOT_BLOCK_SIZE-1;
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > grid_cols-1) ? HOTSPOT_BLOCK_SIZE-1-(blkXmax-grid_cols+1) : HOTSPOT_BLOCK_SIZE-1;
 
        int N = ty-1;
        int S = ty+1;
        int W = tx-1;
        int E = tx+1;
 
        N = (N < validYmin) ? validYmin : N;
        S = (S > validYmax) ? validYmax : S;
        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;
 
        bool computed;
        for (int i=0; i<iteration ; i++){
            computed = false;
            if( IN_RANGE(tx, i+1, HOTSPOT_BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(ty, i+1, HOTSPOT_BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(tx, validXmin, validXmax) && \
                  IN_RANGE(ty, validYmin, validYmax) ) {
                  computed = true;
                  temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] +
                     (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0*temp_on_cuda[ty][tx]) * Ry_1 +
                     (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0*temp_on_cuda[ty][tx]) * Rx_1 +
                     (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
 
            }
            __syncthreads();
            if(i==iteration-1)
                break;
            if(computed)     //Assign the computation range
                temp_on_cuda[ty][tx]= temp_t[ty][tx];
            __syncthreads();
          }
 
      // update the global memory
      // after the last iteration, only threads coordinated within the 
      // small block perform the calculation and switch on ``computed''
      if (computed){
          temp_dst[index]= temp_t[ty][tx];
      }
}
/*
   compute N time steps
*/
 
int compute_tran_temp(float *MatrixPower,float *MatrixTemp[2], int col, int row, \
        int total_iterations, int num_iterations, int blockCols, int blockRows, int borderCols, int borderRows)
{
        dim3 dimBlock(HOTSPOT_BLOCK_SIZE, HOTSPOT_BLOCK_SIZE);
        dim3 dimGrid(blockCols, blockRows);
     
    float grid_height = chip_height / row;
    float grid_width = chip_width / col;
     
    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    float Rz = t_chip / (K_SI * grid_height * grid_width);
     
    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float step = PRECISION / max_slope;
    float t;
        float time_elapsed;
    time_elapsed=0.001;
 
        int src = 1, dst = 0;
 
    for (t = 0; t < total_iterations; t+=num_iterations) {
            int temp = src;
            src = dst;
            dst = temp;
            calculate_temp<<<dimGrid, dimBlock>>>(MIN(num_iterations, total_iterations-t), MatrixPower,MatrixTemp[src],MatrixTemp[dst],\
        col,row,borderCols, borderRows, Cap,Rx,Ry,Rz,step,time_elapsed);
    }
        return dst;
}

/* Reference code: rodinia_3.1/cuda/hotspot/hotspot.cu */
void HLOPGpu::hotspot_2d(KernelParams& kernel_params, void** input, void** output){

    int dim = kernel_params.params.get_kernel_size();
    int grid_rows = dim;
    int grid_cols = dim;
    int size = dim * dim;

    /* some constants */
    int total_iterations = 1;
    int pyramid_height = 1;

    /* pyramid parameters */
    # define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline
    int borderCols = (pyramid_height)*EXPAND_RATE/2;
    int borderRows = (pyramid_height)*EXPAND_RATE/2;
    int smallBlockCol = HOTSPOT_BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int smallBlockRow = HOTSPOT_BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
    int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);

    /* host pointers */
    float* host_float_ptr = reinterpret_cast<float*>(*input);
    float* FilesavingTemp = host_float_ptr;
    float* FilesavingPower = &host_float_ptr[size];
    float* MatrixOut = reinterpret_cast<float*>(*output);

    /* device pointers */
    float *MatrixTemp[2], *MatrixPower;
    cudaMalloc((void**)&MatrixTemp[0], sizeof(float)*size);
    cudaMalloc((void**)&MatrixTemp[1], sizeof(float)*size);
    cudaMemcpy(MatrixTemp[0], FilesavingTemp, sizeof(float)*size, cudaMemcpyHostToDevice);
 
    cudaMalloc((void**)&MatrixPower, sizeof(float)*size);
    cudaMemcpy(MatrixPower, FilesavingPower, sizeof(float)*size, cudaMemcpyHostToDevice);
    //printf("Start computing the transient temperature\n");
    int ret = compute_tran_temp(MatrixPower,MatrixTemp,grid_cols,grid_rows, \
     total_iterations,pyramid_height, blockCols, blockRows, borderCols, borderRows);
    //printf("Ending simulation\n");
    cudaMemcpy(MatrixOut, MatrixTemp[ret], sizeof(float)*size, cudaMemcpyDeviceToHost);
 
    //writeoutput(MatrixOut,grid_rows, grid_cols, ofile);
 
    cudaFree(MatrixPower);
    cudaFree(MatrixTemp[0]);
    cudaFree(MatrixTemp[1]);
}

typedef struct Color {
    unsigned int r, g, b; 
} Color;

void init_means(Color means[], unsigned char *im, int Size_row, int N_colors, int Size) {
    int r;
    int i;
    srand(2023);
    for (i = 0; i < N_colors; ++i) {
        r = rand() % Size;
        int index = (r*3/Size_row) * Size_row + ((r*3)%Size_row);
        means[i].r = im[index+2];
        means[i].g = im[index+1];
        means[i].b = im[index];
    }
}

__global__ void find_best_mean_par(Color means[], int assigns[], unsigned char *im, int N, int ncolors, int Size_row) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        int j;
        int index = (id*3/Size_row) * Size_row + ((id*3)%Size_row);
        int dist_min = -1;
        int dist_act, assign;
        for (j = 0; j < ncolors; ++j) {
            dist_act = (im[index+2] - means[j].r)*(im[index+2] - means[j].r) + (im[index+1] - means[j].g)*(im[index+1] - means[j].g) + (im[index] - means[j].b)*(im[index] - means[j].b);
            if (dist_min == -1 || dist_act < dist_min) {
                dist_min = dist_act;
                assign = j;
            }
        }
        assigns[id] = assign;
    }
}

__global__ void matrix_reduction_count(int counts[], int assigns[], unsigned char *im, int Size_row, int Size, int N_colors) {
    extern __shared__ unsigned int shared[];
 
    unsigned int tid = threadIdx.x;
    unsigned int id = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
 
    //init shared
    for (int j = 0; j < N_colors; ++j) {
        unsigned int aux = 0;
 
        if (j == assigns[id]) {
            aux += 1;
        }
 
        if (j == assigns[id + blockDim.x]) {
            aux += 1;
        }
 
        shared[tid*N_colors + j] = aux;
    }
    __syncthreads();

    unsigned int s;
    for(s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            for (int j = 0; j < N_colors; ++j) {
                shared[tid*N_colors + j] += shared[(tid + s)*N_colors + j];
            }
        }
        __syncthreads();
    }

    //copiar valors:
    if (tid == 0) {
       for (int j = 0; j < N_colors; ++j) {
            counts[blockIdx.x*N_colors + j] = shared[j];
       }
    }
}

__global__ void matrix_reduction_color(Color new_means[], int assigns[], unsigned char *im, int Size_row, int Size, int N_colors, int offset) {
    extern __shared__ unsigned int shared[];
 
    unsigned int tid = threadIdx.x;
    unsigned int id = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
 
    //init shared
    for (int j = 0; j < N_colors; ++j) {
        unsigned int aux = 0;
 
        if (id < Size && j == assigns[id]) {
            int index = (id*3/Size_row) * Size_row + ((id*3)%Size_row);
            aux += im[index];
        }
 
        if (id + blockDim.x < Size && j == assigns[id + blockDim.x]) {
            int index = ((id + blockDim.x)*3/Size_row) * Size_row + (((id + blockDim.x)*3)%Size_row);
            aux += im[index+offset];
        }
 
        shared[tid*N_colors + j] = aux;
 
    }
 
    __syncthreads();
 
    //reduccio
    unsigned int s;
    for(s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            for (int j = 0; j < N_colors; ++j) {
                shared[tid*N_colors + j] += shared[(tid + s)*N_colors + j];
            }
        }
        __syncthreads();
    }
    
    //copiar valors:
    if (tid == 0) {
        for (int j = 0; j < N_colors; ++j) {
            if (offset == 2) new_means[blockIdx.x*N_colors + j].r = shared[j];
            else if (offset == 1) new_means[blockIdx.x*N_colors + j].g = shared[j];
            else new_means[blockIdx.x*N_colors + j].b = shared[j];
        }
    }
}

__global__ void matrix_reduction_count_2(int counts_2[], int counts[], int Size_row, int Size, int N_colors) {
    extern __shared__ unsigned int shared[];
 
    unsigned int tid = threadIdx.x;
    unsigned int id = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
 
    //init shared
    for (int j = 0; j < N_colors; ++j) {
        shared[tid*N_colors + j] = counts[id*N_colors + j] + counts[((id + blockDim.x) * N_colors) + j];
    }
 
    __syncthreads();
 
    //reduccio
    unsigned int s;
    for(s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            for (int j = 0; j < N_colors; ++j) {
               shared[tid*N_colors + j] += shared[(tid + s)*N_colors + j];
            }
        }
        __syncthreads();
    }

    //copiar valors:
    if (tid == 0) {
       for (int j = 0; j < N_colors; ++j) {
            counts_2[blockIdx.x*N_colors + j] = shared[j];
        }
    }
}

__global__ void matrix_reduction_color_2(Color new_means_2[], Color new_means[], int Size_row, int Size, int N_colors, int offset) {
    extern __shared__ unsigned int shared[];
 
    unsigned int tid = threadIdx.x;
    unsigned int id = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
 
    //init shared
    for (int j = 0; j < N_colors; ++j) {
        if (offset == 2)         shared[tid*N_colors + j] = new_means[id*N_colors + j].r + new_means[(id + blockDim.x) *N_colors + j].r;
        else if (offset == 1)    shared[tid*N_colors + j] = new_means[id*N_colors + j].g + new_means[(id + blockDim.x) * N_colors + j].g;
        else                     shared[tid*N_colors + j] = new_means[id*N_colors + j].b + new_means[(id + blockDim.x) *N_colors + j].b;

    }
 
    __syncthreads();

    //reduccio
    unsigned int s;
    for(s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            for (int j = 0; j < N_colors; ++j) {
                shared[tid*N_colors + j] += shared[(tid + s)*N_colors + j];
            }
        }
        __syncthreads();
    }

    //copiar valors:
    if (tid == 0) {
        for (int j = 0; j < N_colors; ++j) {
            if (offset == 2) new_means_2[blockIdx.x*N_colors + j].r = shared[j];
            else if (offset == 1) new_means_2[blockIdx.x*N_colors + j].g = shared[j];
            else new_means_2[blockIdx.x*N_colors + j].b = shared[j];
        }
    }
}

__global__ void divide_sums_by_counts_par(Color means_device[], int N_colors, Color new_means[], int counts[]) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N_colors) {
        //Turn 0/0 into 0/1 to avoid zero division.
        if(counts[id] == 0) counts[id] = 1;
        means_device[id].r = new_means[id].r / counts[id];
        means_device[id].g = new_means[id].g / counts[id];
        means_device[id].b = new_means[id].b / counts[id];
    }
}

__global__ void assign_colors_par(Color means[], int assigns[], unsigned char *im, int Size_row, int Size, int N_colors) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < Size) {
        int index = (id*3/Size_row) * Size_row + ((id*3)%Size_row);
        unsigned char step = floor(256./(N_colors-1));
        //im[index]=means[assigns[id]].b;
        //im[index + 1]=means[assigns[id]].g;
        //im[index + 2]=means[assigns[id]].r;
        im[index]= min(assigns[id] * step, 255);
        //im[index + 1]=means[assigns[id]].g;
        //im[index + 2]=means[assigns[id]].r;
    }
}

void HLOPGpu::kmeans_2d(KernelParams& kernel_params, void** input, void** output){

    //for(int i = 0 ; i < 100 ; i++){
    //    std::cout << __func__ << ": input: " << (unsigned)((uint8_t*)(*input))[i] << std::endl;
    //}
    
    int N_iterations = 1;
    unsigned int nThreads = 1024; //THREADS;
    int width = kernel_params.params.get_kernel_size();
    int height = kernel_params.params.get_kernel_size();
    int Size = kernel_params.params.get_kernel_size() * kernel_params.params.get_kernel_size();
    unsigned int nBlocks = (Size + nThreads - 1)/nThreads;
    int N_colors = 4;

    //printf("nBlocks: %d, Size: %d, nThreads: %d\n", nBlocks, Size, nThreads);

    dim3 dimGrid(nBlocks, 1, 1);
    dim3 dimBlock(nThreads, 1, 1);
 
    unsigned int nBlocksMeans = (N_colors + nThreads - 1)/nThreads;
 
    dim3 dimGridMeans(nBlocksMeans, 1, 1);

    //obtenir memoria HOST:
    Color *means_host;
    means_host = (Color*) malloc(N_colors*sizeof(Color));
    int *counts_host;
    counts_host = (int*) malloc(sizeof(int) * N_colors);
 
    Color *means_host_red;
    means_host_red = (Color*) malloc((nBlocks/(4*nThreads)) * N_colors*sizeof(Color));
    int *counts_host_red;
    counts_host_red = (int*) malloc((nBlocks/(4*nThreads)) * sizeof(int) * N_colors);
 
    //inicialitzar means:
    unsigned char* im_host = (uint8_t*) malloc(width * height * 3 * sizeof(uint8_t));  //(uint8_t*)*input;
    unsigned char* out_tmp = (uint8_t*) malloc(width * height * 3 * sizeof(uint8_t));  //(uint8_t*)*input;
    for(int i = 0 ; i < width ; i++){
        for(int j = 0 ; j < height ; j++){
            im_host[(i*height+j)*3] = ((uint8_t*)*input)[i*height+j];
            im_host[(i*height+j)*3+1] = 0;//((uint8_t*)*input)[i*height+j];
            im_host[(i*height+j)*3+2] = 0;//((uint8_t*)*input)[i*height+j];
        }
    }

    int Size_row = ((kernel_params.params.get_kernel_size()*3 + 3) / 4 ) * 4;
    init_means(means_host, im_host, Size_row, N_colors, Size);
   
    for(int i = 0 ; i < N_colors ; i++){
        printf("init means_host[%d]: %d\n", i, means_host[i].g);
    }

    //obtenir memoria DEVICE:
    Color *means_device;
    Color *new_means;
    int *counts;
    Color *new_means_2;
    int *counts_2;
 
    int *assigns;
    int *assigns_host = (int*) malloc(Size*sizeof(int));
    unsigned char *im_device;

    cudaMalloc((Color**)&means_device, N_colors*sizeof(Color));

    cudaMalloc((Color**)&new_means, (nBlocks/(nThreads)) * N_colors*sizeof(Color));
    cudaMalloc((int**)&counts, (nBlocks/(nThreads)) * N_colors * sizeof (int));

    cudaMalloc((Color**)&new_means_2, (nBlocks/(2*nThreads)) * N_colors*sizeof(Color));
    cudaMalloc((int**)&counts_2, (nBlocks/(2*nThreads)) * N_colors * sizeof (int));

    int imgSize = ((width*3 + 3) / 4 ) * 4 * height;

    cudaMalloc((int**)&assigns, Size*sizeof(int));
    cudaMalloc((unsigned char**)&im_device,  imgSize * sizeof(unsigned char));

    //copiar dades al device:
    cudaMemcpy(im_device, im_host, imgSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(means_device, means_host, N_colors*sizeof(Color), cudaMemcpyHostToDevice);
    
    int shared_memory_size = N_colors * nThreads * sizeof(unsigned int);
    
    for(int it = 0 ; it < N_iterations ; ++it){
        //set counts and new_means to 0
        cudaMemset (counts, 0, nBlocks/(nThreads) * sizeof (int) * N_colors);
        cudaMemset (new_means, 0, nBlocks/(nThreads)* sizeof (Color) * N_colors);
 
        //for each pixel find the best mean.
        find_best_mean_par<<<dimGrid, dimBlock>>>(means_device, assigns, im_device, Size, N_colors, Size_row);

        cudaDeviceSynchronize();

        matrix_reduction_count<<<nBlocks/(2*nThreads), dimBlock, shared_memory_size>>>(counts, assigns, im_device, Size_row, Size, N_colors);
        matrix_reduction_color<<<nBlocks/(2*nThreads), dimBlock, shared_memory_size>>>(new_means, assigns, im_device, Size_row, Size, N_colors, 2);
        matrix_reduction_color<<<nBlocks/(2*nThreads), dimBlock, shared_memory_size>>>(new_means, assigns, im_device, Size_row, Size, N_colors, 1);
        matrix_reduction_color<<<nBlocks/(2*nThreads), dimBlock, shared_memory_size>>>(new_means, assigns, im_device, Size_row, Size, N_colors, 0);

        cudaDeviceSynchronize();

        //volmemos a hacer otra reduccion
        matrix_reduction_count_2<<<nBlocks/(4*nThreads), dimBlock, shared_memory_size>>>(counts_2, counts, Size_row, Size, N_colors);
        matrix_reduction_color_2<<<nBlocks/(4*nThreads), dimBlock, shared_memory_size>>>(new_means_2, new_means, Size_row, Size, N_colors, 2);
        matrix_reduction_color_2<<<nBlocks/(4*nThreads), dimBlock, shared_memory_size>>>(new_means_2, new_means, Size_row, Size, N_colors, 1);
        matrix_reduction_color_2<<<nBlocks/(4*nThreads), dimBlock, shared_memory_size>>>(new_means_2, new_means, Size_row, Size, N_colors, 0);

        cudaMemcpy(means_host_red, new_means_2, (nBlocks/(4*nThreads)) * N_colors * sizeof(Color), cudaMemcpyDeviceToHost);
        cudaMemcpy(counts_host_red, counts_2, (nBlocks/(4*nThreads)) * N_colors * sizeof(int), cudaMemcpyDeviceToHost);
 
        memset(counts_host, 0, sizeof (int) * N_colors);
        memset(means_host, 0, sizeof (Color) * N_colors);
 
        int i, j;
        //std::cout << __func__ << ": nBlocks: " << nBlocks << ", nThreads: " << nThreads << std::endl;
        for (i = 0; i < nBlocks/(4*nThreads); ++i) {
            for (j = 0; j < N_colors; ++j) {
                counts_host[j] += counts_host_red[i*N_colors + j];
                means_host[j].r += means_host_red[i*N_colors + j].r;
                means_host[j].g += means_host_red[i*N_colors + j].g;
                means_host[j].b += means_host_red[i*N_colors + j].b;
                std::cout << __func__ << ": i: " << i << ",j: " << j << ", means_host[" << j << "]: " << (unsigned)means_host[j].g << ", counts_host: " << counts_host[j] << std::endl;
            }
        }

        //aqui tenemos los vectores finales ya reducidos
        cudaMemcpy(new_means, means_host, N_colors * sizeof(Color), cudaMemcpyHostToDevice);
        cudaMemcpy(counts, counts_host, N_colors * sizeof(int), cudaMemcpyHostToDevice);

        divide_sums_by_counts_par<<<dimGridMeans, dimBlock>>>(means_device, N_colors, new_means, counts);

        cudaDeviceSynchronize();
    }

    //assignem colors:
    assign_colors_par<<<dimGrid, dimBlock>>>(means_device, assigns, im_device, Size_row, Size, N_colors);
        
    cudaMemcpy(assigns_host, assigns, Size * sizeof(int), cudaMemcpyDeviceToHost);
 
    //copy to host:
    cudaMemcpy(out_tmp, im_device, imgSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    for(int i = 0 ; i < width ; i++){
        for(int j = 0 ; j < height ; j++){
            ((uint8_t*)*output)[i*height+j] = out_tmp[(i*height+j)*3];
                                              //sqrt((out_tmp[(i*height+j)*3]   * out_tmp[(i*height+j)*3] +
                                              //out_tmp[(i*height+j)*3+1] * out_tmp[(i*height+j)*3+1] +
                                              //out_tmp[(i*height+j)*3+2] * out_tmp[(i*height+j)*3+2]) / 3.) ;
        }
    }
    
    for(int i = 0 ; i < 100 ; i++){
        std::cout << __func__ << ": im_host: " << (unsigned)((uint8_t*)(im_host))[i*3] << std::endl;
    }

    for(int i = 0 ; i < 100 ; i++){
        std::cout << __func__ << ": out_tmp: " << (unsigned)((uint8_t*)(out_tmp))[i*3] << std::endl;
    }

    for(int i = 0 ; i < Size ; i++){
        std::cout << __func__ << ": assigns_host: " << assigns_host[i*3] << std::endl;
    }
    //alliberar memoria DEVICE:
    free(im_host);
    free(out_tmp);

    cudaFree(means_device);
    cudaFree(new_means);
    cudaFree(new_means_2);
    cudaFree(assigns);
    cudaFree(im_device);
    cudaFree(counts);
    cudaFree(counts_2);

}

void HLOPGpu::laplacian_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img){
    auto laplacian = cuda::createLaplacianFilter(in_img.type(), in_img.type(), 3/*kernel size*/, 1/*scale*/, BORDER_DEFAULT);
    laplacian->apply(in_img, out_img);
    cuda::abs(out_img, out_img);
}

void HLOPGpu::mean_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img){
    auto median = cuda::createBoxFilter(in_img.type(), in_img.type(), Size(3, 3),     Point(-1, -1), BORDER_DEFAULT);
    median->apply(in_img, out_img);
}

void HLOPGpu::minimum_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img){
    out_img = in_img;
}

void HLOPGpu::sobel_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img){
    cuda::GpuMat grad_x, grad_y;
    cuda::GpuMat abs_grad_x, abs_grad_y;

    int ddepth = CV_32F;
    auto sobel_dx = cuda::createSobelFilter(in_img.type(), ddepth, 1, 0, 3);
    auto sobel_dy = cuda::createSobelFilter(in_img.type(), ddepth, 0, 1, 3);
 
    sobel_dx->apply(in_img, grad_x);
    sobel_dy->apply(in_img, grad_y);
 
    cuda::abs(grad_x, abs_grad_x);
    cuda::abs(grad_y, abs_grad_y);
  
    cuda::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, out_img);
}

void HLOPGpu::histogram_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img){
    cuda::calcHist(in_img, out_img);
}

void HLOPGpu::srad_2d(KernelParams& kernel_params, void** input, void** output){
    int rows = kernel_params.params.get_kernel_size();
    int cols = kernel_params.params.get_kernel_size();
    int size_I, size_R, niter = 1, iter;
    float *I, *J, lambda=0.5, q0sqr, sum, sum2, tmp, meanROI, varROI;

    float *J_cuda;
    float *C_cuda;
    float *E_C, *W_C, *N_C, *S_C;

    unsigned int r1 = 0, r2 = rows-1, c1 = 0, c2 = cols-1; // need init
    float *c;

    size_I = cols * rows;
    size_R = (r1-r1+1)*(c2-c1+1);
 
    I = (float*)*input;
    J = (float*)*output;
    c = (float *)malloc(sizeof(float)* size_I);

    //Allocate device memory
    cudaMalloc((void**)& J_cuda, sizeof(float)* size_I);
    cudaMalloc((void**)& C_cuda, sizeof(float)* size_I);
    cudaMalloc((void**)& E_C, sizeof(float)* size_I);
    cudaMalloc((void**)& W_C, sizeof(float)* size_I);
    cudaMalloc((void**)& S_C, sizeof(float)* size_I);
    cudaMalloc((void**)& N_C, sizeof(float)* size_I);

    for (int k = 0;  k < size_I; k++ ) {
        J[k] = (float)exp(I[k]) ;
    }

    for(iter=0; iter < niter ; iter++){
        sum=0; sum2=0;
        for (int i=r1; i<=r2; i++) {
            for (int j=c1; j<=c2; j++) {
                tmp   = J[i * cols + j];
                sum  += tmp ;
                sum2 += tmp*tmp;
            }
        }
        meanROI = sum / size_R;
        varROI  = (sum2 / size_R) - meanROI*meanROI;
        q0sqr   = varROI / (meanROI*meanROI);

        //Currently the input size must be divided by 16 - the block size
        int block_x = cols/SRAD_BLOCK_SIZE ;
        int block_y = rows/SRAD_BLOCK_SIZE ;
 
        dim3 dimBlock(SRAD_BLOCK_SIZE, SRAD_BLOCK_SIZE);
        dim3 dimGrid(block_x , block_y);
 
        //Copy data from main memory to device memory
        cudaMemcpy(J_cuda, J, sizeof(float) * size_I, cudaMemcpyHostToDevice);
 
        //Run kernels
        srad_cuda_1<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda,     cols, rows, q0sqr);
        srad_cuda_2<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda,     cols, rows, lambda, q0sqr);
 
        //Copy data from device memory to main memory
        cudaMemcpy(J, J_cuda, sizeof(float) * size_I, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    //cudaThreadSynchronize();

    cudaFree(C_cuda);
    cudaFree(J_cuda);
    cudaFree(E_C);
    cudaFree(W_C);
    cudaFree(N_C);
    cudaFree(S_C);
    free(c);
}

inline void fdwt(float *in, float *out, int width, int height, int levels){
    dwt_cuda::fdwt97(in, out, width, height, levels);
}

inline void rdwt(float *in, float *out, int width, int height, int levels){
    dwt_cuda::rdwt97(in, out, width, height, levels);
}

void HLOPGpu::dwt_2d(KernelParams& kernel_params, void** input, void** output){
    int width  = kernel_params.params.get_kernel_size();
    int height = kernel_params.params.get_kernel_size();
    int componentSize = width * height * sizeof(float);

    float* c_r; // device input
    float* c_r_mid;  // device output
    float* c_r_out;  // device output
    //float* backup ;
    
    cudaMalloc((void**)&(c_r), componentSize); //< R, aligned component size
    cudaMemset(c_r, 0, componentSize);

    cudaMalloc((void**)&c_r_mid, componentSize); //< aligned component size
    cudaMemset(c_r_mid, 0, componentSize);
    
    cudaMalloc((void**)&c_r_out, componentSize); //< aligned component size
    cudaMemset(c_r_out, 0, componentSize);

    cudaMemcpy(c_r, (float*)*input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    
//    fdwt(c_r,     c_r_out, width, height, 1/*stages*/);
    fdwt(c_r,     c_r_mid, width, height, 1/*stages*/);
    rdwt(c_r_mid, c_r_out, width, height, 1/*stages*/);

    cudaMemcpy((float*)*output, c_r_out, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(c_r);
    cudaFree(c_r_mid);
    cudaFree(c_r_out);
}

