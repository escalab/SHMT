#include <math.h>
#include <string>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <cuda_runtime.h>
#include "dwt.h"
#include "common.h"

inline void fdwt(float *in, float *out, int width, int height, int levels){
    dwt_cuda::fdwt97(in, out, width, height, levels);
}

inline void rdwt(float *in, float *out, int width, int height, int levels){
    dwt_cuda::rdwt97(in, out, width, height, levels);
}

extern "C" void dwt_2d(int row, int col, float* input, float* output){
    int width  = row;
    int height = col;
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

    cudaMemcpy(c_r, (float*)input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    
//    fdwt(c_r,     c_r_out, width, height, 1/*stages*/);
    fdwt(c_r,     c_r_mid, width, height, 1/*stages*/);
    rdwt(c_r_mid, c_r_out, width, height, 1/*stages*/);

    cudaMemcpy((float*)output, c_r_out, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(c_r);
    cudaFree(c_r_mid);
    cudaFree(c_r_out);
}

