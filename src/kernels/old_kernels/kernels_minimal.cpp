#include <string>
#include <stdio.h>
#include "kernels_cpu.h"
#include "kernels_gpu.h"

/* A dummy kernel for testing only. */
void CpuKernel::minimum_2d(const Mat in_img, Mat& out_img){
    out_img = in_img;
}

void GpuKernel::minimum_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img){
    out_img = in_img;
}
