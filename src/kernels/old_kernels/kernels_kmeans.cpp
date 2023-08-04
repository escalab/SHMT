#include "kernels_cpu.h"
#include "kernels_gpu.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

void CpuKernel::kmeans_2d(const Mat in_img, Mat& out_img){
    int k = 4;
    std::vector<int> labels;
    cv::Mat1f centers;
    unsigned int size = in_img.rows * in_img.cols;
    cv::Mat in_tmp = in_img.reshape(1, size);
    in_tmp.convertTo(in_tmp, CV_32F);
    cv::kmeans(in_tmp, // kmeans only takes CV_32F data type 
               k, 
               labels, 
               cv::TermCriteria(TermCriteria::MAX_ITER/*|TermCriteria::EPS*/, 
                                10, // max iteration 
                                1.0), // epsilon
               3, // attempts
               cv::KMEANS_PP_CENTERS,
               centers);
    for (unsigned int i = 0; i < size; i++) {
        in_tmp.at<float>(i) = centers(labels[i]);
    }
    out_img = in_tmp.reshape(1, in_img.rows);
    out_img.convertTo(out_img, CV_8U);
}

void GpuKernel::kmeans_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img){
}
