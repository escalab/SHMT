#include <iostream>
#include "shmt.h"

std::vector<DeviceType> VOPS::run_kernel_on_single_device(
        const std::string& mode,
        Params params,
        void* input,
        void* output,
        TimeBreakDown* t){
    std::vector<DeviceType> ret;
    HLOPBase* kernel = NULL;
    if(mode == "cpu"){
        kernel = new HLOPCpu(params, input, output);
        ret.push_back(cpu);
    }else if(mode == "gpu"){
        kernel = new HLOPGpu(params, input, output);
        ret.push_back(gpu);
    }else if(mode == "tpu"){
        kernel = new HLOPTpu(params, input, output);
        ret.push_back(tpu);
    }else{
        std::cout << __func__ << ": undefined execution mode: " << mode
                  << ", execution is skipped." << std::endl;
    }
    // input array conversion from void* input
    t->input_time_ms = kernel->input_conversion();
 
    // Actual kernel call
    t->kernel_time_ms = kernel->run_kernel(params.iter);
 
    // output array conversion back to void* output
    t->output_time_ms = kernel->output_conversion();
/*
    std::cout << __func__ << ": "
              << input_time_ms << " (ms), "
              << kernel_time_ms << " (ms), "
              << output_time_ms << std::endl;
*/
    delete kernel;
    return ret;
}

std::vector<DeviceType> VOPS::run_kernel_partition(
        const std::string& mode,
        Params params,
        void* input,
        void* output,
        TimeBreakDown* t){
    PartitionRuntime* p_run = new PartitionRuntime(params,
                                                   mode,
                                                   input,
                                                   output);
    t->input_time_ms = p_run->prepare_partitions();
 
    // Actual kernel call
    t->kernel_time_ms = p_run->run_partitions();
 
    t->output_time_ms = p_run->transform_output();
 
    //p_run->show_device_sequence();
    std::vector<DeviceType> ret = p_run->get_device_sequence();
/*
    std::cout << __func__ << ": "
              << input_time_ms << " (ms), "
              << kernel_time_ms << " (ms), "
              << output_time_ms << std::endl;
 */
    delete p_run;
    return ret;
}

std::vector<DeviceType> VOPS::run_kernel(const std::string& mode,
                                   Params& params,
                                   void* input,
                                   void* output,
                                   TimeBreakDown* t){
//    params.set_kernel_size(this->blk_size[params.app_name]);
    std::vector<DeviceType> ret;
    if(mode == "cpu" || mode == "gpu" || mode == "tpu"){
        ret = this->run_kernel_on_single_device(mode,
                                                params,
                                                input,
                                                output,
                                                t);
    }else{
        ret = this->run_kernel_partition(mode,
                                         params,
                                         input,
                                         output,
                                         t);
    }
    return ret;
}
