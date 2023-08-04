#include <string>
#include <stdio.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include "kernels_cpu.h"
#include "kernels_gpu.h"

void CpuKernel::laplacian_2d(const Mat in_img, Mat& out_img){
    int ddepth = CV_32F;
    Laplacian(in_img, out_img, ddepth, 3/*kernel size*/, 1/*scale*/, 0/*delta*/, BORDER_DEFAULT);
    convertScaleAbs(out_img, out_img);
}

void GpuKernel::laplacian_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img){
    auto laplacian = cuda::createLaplacianFilter(in_img.type(), in_img.type(), 3/*kernel size*/, 1/*scale*/, BORDER_DEFAULT);
    laplacian->apply(in_img, out_img);
    cuda::abs(out_img, out_img);
}
