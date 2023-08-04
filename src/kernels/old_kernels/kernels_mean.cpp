#include <string>
#include <stdio.h>
#include <opencv2/cudafilters.hpp> // create[XXX]Filter()
#include "kernels_cpu.h"
#include "kernels_gpu.h"

void CpuKernel::mean_2d(const Mat in_img, Mat& out_img){
    blur(in_img, out_img, Size(3, 3), Point(-1, -1), BORDER_DEFAULT);
}

void GpuKernel::mean_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img){
    auto median = cuda::createBoxFilter(in_img.type(), in_img.type(), Size(3, 3),     Point(-1, -1), BORDER_DEFAULT);
    median->apply(in_img, out_img);
}
