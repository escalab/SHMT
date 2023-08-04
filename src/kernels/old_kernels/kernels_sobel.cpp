#include <string>
#include <stdio.h>
#include <opencv2/cudaarithm.hpp> // addWeighted()
#include <opencv2/cudafilters.hpp> // create[XXX]Filter()
#include "kernels_cpu.h"
#include "kernels_gpu.h"

void CpuKernel::sobel_2d(const Mat in_img, Mat& out_img){
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    int ddepth = CV_32F; // CV_8U, CV_16S, CV_16U, CV_32F, CV_64F
    Sobel(in_img, grad_x, ddepth, 1, 0, 3/*ksize*/, 1/*scale*/, 0/*delta*/, BORDER_DEFAULT);
    Sobel(in_img, grad_y, ddepth, 0, 1, 3/*ksize*/, 1/*scale*/, 0/*delta*/, BORDER_DEFAULT);

    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, out_img);
}
 
void GpuKernel::sobel_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img){

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
