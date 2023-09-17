#include <cassert>
#include "params.h"

Params::Params(
    std::string app_name,
    int problem_size,
    int block_size,
    bool tiling_mode,
    std::string test_mode,
    std::string input_data_path){

    this->app_name        = app_name;
    this->problem_size    = problem_size;
    this->block_size      = block_size;
    this->tiling_mode     = tiling_mode,
    assert(test_mode == "performance" || test_mode == "quality");
    this->iter            = (test_mode == "performance")?
	    			100:((test_mode == "quality")?1:0);
    if(app_name == "fft_2d")
	    this->iter = 1; // to avoid OOM
    if(app_name == "blackscholes_2d")
        this->iter = 30; // to avoid hanging
    this->input_data_path = input_data_path;

    assert(problem_size >= block_size);
    // A temporary aligned design, can be released later
    assert(problem_size % block_size == 0);
    this->row_cnt = problem_size / block_size; 
    this->col_cnt = problem_size / block_size; 
    this->block_cnt = this->row_cnt * this->col_cnt;
}

void Params::set_tiling_mode(bool flag){
    this->tiling_mode = flag;
}

void Params::set_kernel_size(unsigned int size){
    this->block_size = size;
}

bool Params::get_tiling_mode(){
    return this->tiling_mode;
}

unsigned int Params::get_kernel_size(){
    return (this->tiling_mode)?this->block_size:this->problem_size;
}

unsigned int Params::get_row_cnt(){
    return (this->tiling_mode)?this->row_cnt:1;
}

unsigned int Params::get_col_cnt(){
    return (this->tiling_mode)?this->col_cnt:1;
}

unsigned int Params::get_block_cnt(){
    return (this->tiling_mode)?this->block_cnt:1;
}
