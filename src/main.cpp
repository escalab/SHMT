#include <ctime>
#include <fstream>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "shmt.h"
   
int main(int argc, char* argv[]){
    if(argc < 7){
        std::cout << "Usage: " << argv[0] 
                  << " <application name>" // kernel's name
                  << " <problem_size>" // given problem size
                  << " <block_size>" // desired blocking size (effective only if tiling mode(s) is chosen.)
                  << " <test_mode>" // perf or quality
                  << " <baseline mode>"
                  << " <proposed mode>"
                  << std::endl;
        return 0;
    }else{
        // print program arguments
        for(int i = 0 ; i < argc ; i++){
            std::cout << argv[i] << " ";
        }
        std::cout << std::endl;
    }

    // program arguments assignment
    int idx = 1;
    std::string app_name = argv[idx++];
    int problem_size     = atoi(argv[idx++]);
    int block_size       = atoi(argv[idx++]);
    //int iter             = atoi(argv[idx++]);
    std::string test_mode     = argv[idx++];
    std::string baseline_mode = argv[idx++];
    std::string proposed_mode = argv[idx++];
    
    std::string testing_img_path = 
        (argc == 8)?argv[idx++]:"../data/lena_gray_2Kx2K.bmp";
    std::string testing_img_file_name = 
        testing_img_path.substr(testing_img_path.find_last_of("/") + 1);
    
    void* input_array = NULL;
    void* output_array_baseline = NULL;
    void* output_array_proposed = NULL;

    TimeBreakDown* baseline_time_breakdown = new TimeBreakDown;
    TimeBreakDown* proposed_time_breakdown = new TimeBreakDown;

    std::vector<DeviceType> baseline_device_sequence; // typically will be gpu mode
    std::vector<DeviceType> proposed_device_sequence;

    std::vector<bool> baseline_criticality_sequence;
    std::vector<bool> proposed_criticality_sequence;

    // VOPS
    VOPS baseline_vops;
    VOPS proposed_vops;

    // create parameter instances
    Params baseline_params(app_name,
                           problem_size, 
                           block_size, 
                           false, // default no tiling mode. can be reset anytime later
                           test_mode,
                           testing_img_path); 
    Params proposed_params(app_name,
                           problem_size, 
                           block_size, 
                           false, // default no tiling mode. can be reset anytime later
                           test_mode,
                           testing_img_path);

    /* input/output array allocation and inititalization
        All arrays will be casted to corresponding data type
        depending on application.
     */
    std::cout << "data init..." << std::endl;
    data_initialization(proposed_params, 
                        &input_array,
                        &output_array_baseline,
                        &output_array_proposed);

    // Start to run baseline version of the application's implementation.
    std::cout << "run baseline... " << baseline_mode << std::endl; 
    baseline_device_sequence = baseline_vops.run_kernel(baseline_mode,
                                                        baseline_params,
                                                        input_array,
                                                        output_array_baseline,
                                                        baseline_time_breakdown);
    
    // Start to run proposed version of the application's implementation
    std::cout << "run experiment... " << proposed_mode << std::endl; 
    proposed_device_sequence = proposed_vops.run_kernel(proposed_mode,
                                                        proposed_params,
                                                        input_array,
                                                        output_array_proposed,
                                                        proposed_time_breakdown);

    // convert device sequence type 
    std::vector<int> proposed_device_type;
    for(unsigned int i = 0 ; i < proposed_device_sequence.size() ; i++){
        proposed_device_type.push_back(int(proposed_device_sequence[i]));
    }

    UnifyType* unify_input_type = 
        new UnifyType(baseline_params, input_array);
    UnifyType* unify_baseline_type = 
        new UnifyType(baseline_params, output_array_baseline);    
    UnifyType* unify_proposed_type = 
        new UnifyType(proposed_params, output_array_proposed);    

    // Get quality measurements
    std::cout << "Result evaluating..." << std::endl;
    Quality* quality = new Quality(app_name, 
                                   proposed_params.problem_size, // m
                                   proposed_params.problem_size, // n
                                   proposed_params.problem_size, // ldn
                                   proposed_params.block_size,
                                   proposed_params.block_size,
                                   unify_input_type->float_array,
                                   unify_proposed_type->float_array, 
                                   unify_baseline_type->float_array,
                                   proposed_device_type);
    
    std::cout << "--------------- Summary ---------------" << std::endl;
    if(test_mode == "performance"){
    std::cout << "e2e time: " 
              << baseline_time_breakdown->get_total_time_ms(baseline_params.iter) 
              << " (ms), " 
              << proposed_time_breakdown->get_total_time_ms(proposed_params.iter) 
              << " (ms), speedup: " 
              << baseline_time_breakdown->get_total_time_ms(baseline_params.iter) / 
                 proposed_time_breakdown->get_total_time_ms(proposed_params.iter)  
              << std::endl;
    }
    if(test_mode == "quality")    
    	std::cout << "mape: " << quality->error_rate() << " %" << std::endl;
    std::cout << std::endl;

    delete quality;
    delete baseline_time_breakdown;
    delete proposed_time_breakdown;
    delete unify_baseline_type;
    delete unify_proposed_type;
    return 0;
}
