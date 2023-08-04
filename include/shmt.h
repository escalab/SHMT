#ifndef __VOPS_H__
#define __VOPS_H__
#include <iostream>
#include <algorithm>
#include "types.h" // utils
#include "utils.h" // utils
#include "arrays.h"
#include "params.h"
#include "quality.h"
#include "hlop_cpu.h"
#include "hlop_gpu.h"
#include "hlop_tpu.h"
#include "partition.h"
#include "conversion.h"
#include "performance.h"

class VOPS{
public:
    std::vector<DeviceType> run_kernel(
            const std::string& mode,
            Params& params,
            void* input,
            void* output,
            TimeBreakDown* t);
private:
    std::vector<DeviceType> run_kernel_on_single_device(
            const std::string& mode,
            Params params,
            void* input,
            void* output,
            TimeBreakDown* t);
    std::vector<DeviceType> run_kernel_partition(
            const std::string& mode,
            Params params,
            void* input,
            void* output,
            TimeBreakDown* t);
    std::unordered_map<std::string, unsigned int> blk_size = {
        {"mean_2d",         512},
        {"sobel_2d",        2048},
        {"laplacian_2d",    2048},
        {"fft_2d",          1024},
        {"dct8x8_2d",       2048},
        {"hotspot_2d",      2048},
        {"srad_2d",         512},
        {"dwt_2d",          2048},
        {"blackscholes_2d", 2048},
        {"histogram_2d",    2048}};
};
#endif
