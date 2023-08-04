#ifndef __UTILS_H__
#define __UTILS_H__
#include <chrono>
#include <string.h>
#include <opencv2/opencv.hpp>
#include "types.h"
#include "quality.h"
#include "performance.h"

using namespace cv;

double get_time_ms(timing end, timing start);
void read_img(const std::string file_name, int rows, int cols, Mat& img);
void save_float_image(const std::string file_name, 
                      unsigned int rows, 
                      unsigned int cols,
                      float* img);
void mat2array(Mat img, uint8_t* data);
void mat2array(Mat img, float* data);
void mat2array(cuda::GpuMat img, uint8_t* data);
void mat2array(cuda::GpuMat img, float* data);
void mat2array(cuda::GpuMat img, int* data);
void mat2array_CV_32F2uchar(cuda::GpuMat img, uint8_t* data);

void array2mat(Mat& img, float* data, int rows, int cols);
void array2mat(Mat& img, uint8_t* data, int rows, int cols);
void array2mat(cuda::GpuMat& img, float* data, int rows, int cols);
void array2mat(cuda::GpuMat& img, uint8_t* data, int rows, int cols);
void array2mat_uchar2CV_32F(cuda::GpuMat& img, uint8_t* data, int rows, int cols);

std::string get_edgetpu_kernel_path(std::string app_name, 
                                    int shape0, 
                                    int shape1);
void histogram_matching(void* output_array_baseline,
                        void* output_array_proposed,
                        int rols,
                        int cols,
                        int blk_rows,
                        int blk_cols,
                        std::vector<int> dev_sequence);
void dump_to_csv(std::string log_file_path,
                 std::string input_img_name,
                 std::string app_name,
                 std::string baseline_mode,
                 std::string proposed_mode,
                 unsigned int problem_size,
                 unsigned int block_size,
                 unsigned int iter,
                 Quality* quality, 
                 TimeBreakDown* baseline_time_breakdown,
                 TimeBreakDown* proposed_time_breakdown,
                 std::vector<int> proposed_device_sequence,
                 float saliency_ratio,
                 float protected_saliency_ratio);
#endif

