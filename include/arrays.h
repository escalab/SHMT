#ifndef __ARRAYS_H__
#define __ARRAYS_H__
#include <vector>
#include "params.h"

void array_partition_initialization(Params params, 
                                    bool skip_init,      
                                    void** input, 
                                    std::vector<void*>& input_pars);
void output_array_partition_gathering(Params params, 
                                      void** output, 
                                      std::vector<void*>& output_pars);
void data_initialization(Params params, 
                         void** input_array, 
                         void** output_array_baseline, 
                         void** output_array_proposed);

void array_partition_downsampling(Params params,
                                  bool skip_init,
                                  std::vector<void*> input_pars,
                                  std::vector<void*>& input_sampling_pars);
#endif

