#ifndef __PARTITION_H__
#define __PARTITION_H__
#include <string>
#include <iostream>
#include <pthread.h>
#include "params.h"
#include "hlop_cpu.h"
#include "hlop_gpu.h"
#include "hlop_tpu.h"
#include "concurrentqueue.h"

typedef enum _DeviceType { undefine, cpu, gpu, tpu} DeviceType;

typedef struct {
    HLOPBase* kernel_base = NULL;
    DeviceType device_type;
}GenericKernel;

/*
    structure of each SPMC FIFO node. 
 */
struct node_data{
    GenericKernel* generic_kernel;
    Params params;
    unsigned int block_id;
    unsigned int iter;
};

class PartitionRuntime{
public:
    PartitionRuntime(Params params, 
                     std::string mode, 
                     void* input, 
                     void* output);
    
    ~PartitionRuntime();
    double prepare_partitions();
    double run_partitions();
    double transform_output();
    void show_device_sequence();
    std::vector<DeviceType> get_device_sequence();
    std::vector<bool> get_criticality(){ return this->criticality; };
    
    unsigned int dev_type_cnt = 3; // cpu, gpu and tpu

private:
    /* The sampling pre-processing to determine criticality on tiling blocks. 
        Return is timing overhead in ms.
        This is for acutal run types of sampling
     */
    double run_sampling(SamplingMode mode);

    /*
       Get saliency binary map of input image and assign criticality on each tiling block 
        threshold: 
            A theshold to determine if a tiling block is critical or not by the ratio: 
                    (# of saliency pixels) / (# of total pixels)
        criticality_ratio:
            The upper bound ratio of (# of critical tiling blocks) / (# of tiling blocks).
     */
    double set_criticality_by_saliency(Params params, void** array);

    /* The sampling pre-processing to determine criticality on tiling blocks. 
        Return is timing overhead in ms.
        This is for input stats probing types of sampling
     */
    double run_input_stats_probing(std::string mode, unsigned int num_pixels);
    double run_input_homo_probing(float one_dim_ratio);

    /* The main algorithm to determine tiling tasks to specific device(s). */
    DeviceType mix_policy(unsigned int i);
    
    /* threading for each device type. */
    static void* RunDeviceThread(void* args);

    /* SPMC queue */
    moodycamel::ConcurrentQueue<struct node_data> q;
    
    void create_kernel_by_type(unsigned int block_id, DeviceType device_type);
    
    // For sampling policy use only
    std::vector<Quality> sampling_qualities;
    std::vector<bool> criticality;
    // A helper function to check if mode is criticality mode
    bool is_criticality_mode();

    // A helper function to extract partition mode is string
    std::string get_partition_mode();

    // The main criticality determine function based on sampling qualities.
    void criticality_kernel(Params params, 
                            std::vector<std::pair<int, float>>& order, 
                            float criticality_ratio,
                            std::string mode/*threshold, topK*/);

    /* To determine if each type of devices is static or dynamic
        by setting the the following arrays:
        bool* is_dynamic_device
     */
    void setup_dynamic_devices();
    
    /* To determine if each tiling block is static or dynamic
        by setting the the following arrays:
        bool* is_dynamic_block

        If criticality has length zero, then sampling is off.
     */
    void setup_dynamic_blocks();

    /*
        To indicate if the ith tiling block is under dynamic partitioning mode. 
        Dynamic block:
            a tiling block that will only be assigned to a device until 
            runtime scheduling happens such as SPMC.
        Static block:
            a tiling block that can be assigned to a device statically 
            before execution stage.
     */
    bool* is_dynamic_block;
    
    /*
        To indicate if the ith device participates in SPMC scheduling, 
        which means it's allowed to consume any dynamic tiling block.
     */
    bool* is_dynamic_device;

    unsigned int row_cnt = 1;
    unsigned int col_cnt = 1;
    unsigned int block_cnt = 1; // default no partition
    Params params;
    std::string mode = "cpu_p"; // partition mode, default as cpu_p
    void* input;
    void* output;
    HLOPCpu** cpu_kernels;
    HLOPGpu** gpu_kernels;
    HLOPTpu** tpu_kernels;

    GenericKernel* generic_kernels;
    DeviceType* dev_sequence; // device sequence storing device types of each tiling block
    
    std::vector<void*> input_pars;
    std::vector<void*> output_pars;
};

/*
    pthread arguments
    Each thread here represents one type of device.
 */
struct thread_data{
    PartitionRuntime* p_run_ptr;
    GenericKernel* generic_kernels;
    unsigned int block_cnt;
    unsigned int iter;
    double kernel_ms;
    // The device type the thread is representing.
    DeviceType device_type;
};
#endif
