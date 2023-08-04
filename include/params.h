#ifndef __PARAMS_H__
#define __PARAMS_H__
#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <iostream>
#include <unordered_map>

enum SamplingMode {cv_resize, center_crop, init_crop, random_crop};

class Params{
public:
    Params(std::string app_name="sobel_2d",
          int problem_size=2048,
          int block_size=2048,
          bool tiling_mode = false,
          std::string test_mode = "performance", // perf or quality
          //std::string input_data_path="../data/lena_gray_2Kx2K.bmp"
          std::string input_data_path="../data/super5.png"
          );
    
    void set_tiling_mode(bool);
    bool get_tiling_mode();
    unsigned int get_row_cnt();
    unsigned int get_col_cnt();
    unsigned int get_block_cnt();
    unsigned int get_kernel_size();
    void set_kernel_size(unsigned int size);
    void set_downsampling_rate(float r){this->downsampling_rate = r; };
    float get_downsampling_rate(){ return this->downsampling_rate; };
    void set_sampling_mode(SamplingMode mode){ this->sampling_mode = mode; };
    SamplingMode get_sampling_mode(){ return this->sampling_mode; };

    std::string app_name;
    int problem_size;
    int block_size;
    bool tiling_mode;
    unsigned int iter;
    std::string input_data_path; 

    std::vector<std::string> uint8_t_type_app = {
        "sobel_2d",
        "mean_2d",
        "laplacian_2d",
        "kmeans_2d",
        "histogram_2d"
    };
    
    /* Return maximun percentage of tiling blocks can be protected as critical
        before degrading latency.
     */
    void set_criticality_ratio(float val){ this->criticality_ratio = val; };
    void set_criticality_ratio(){
        unsigned int idx = log2(this->problem_size / 1024);
        assert(this->criticality_ratio_table.find(this->app_name) != this->criticality_ratio_table.end());
        float ratio = this->criticality_ratio_table[this->app_name][idx];// 1. / (this->criticality_ratio_table[this->app_name][idx] + 1.);
        this->criticality_ratio = ratio;
    };
    float get_criticality_ratio(/*std::string app_name, int block_size*/){ return this->criticality_ratio; };
    void set_num_sample_pixels(int v){ this->num_sample_pixels = v; };
    int get_num_sample_pixels(){ return this->num_sample_pixels; };

private:        
    unsigned int row_cnt = 0;
    unsigned int col_cnt = 0;
    unsigned int block_cnt = 0;
    float downsampling_rate = 0.25;
    SamplingMode sampling_mode = center_crop;
    
    float criticality_ratio = 1./3.;
    int num_sample_pixels = 100;

    /* Criticality ratio table is based on real measurement. 
        The value is the latency ratio: edgeTPU over GPU baseline
     */
    std::unordered_map<std::string, std::vector<float>> criticality_ratio_table = {
        {"mean_2d",         {0.265673476182506, 0.658661581323829, 1.53029574244256,  0.8398}},
        {"sobel_2d",        {0.218395820507639, 0.509887148867899, 1.08003270414455,  0.390625}},
        {"laplacian_2d",    {0.269053766577797, 0.741906709445404, 1.43002676010527,  0.75}},
        {"fft_2d",          {0.640955962788173, 0.698180086947237, 0.310684776714147, 0.21875}},
        {"dct8x8_2d",       {0.503233693347007, 0.435987033037118, 0.525195542239914, 0.48046875}},
        {"hotspot_2d",      {1.29435361772579,  1.72613253937488,  1.71823889626435,  0.5859375}},
        {"srad_2d",         {0.700557163661798, 0.609395769414589, 0.466719978570055, 0.54296875}},
        {"dwt_2d",          {3.20018280519802,  3.81926113760053,  3.2507056318647,   0.5859375}},
        {"blackscholes_2d", {1.19454239762865,  2.65023275081232,  4.05394756070832,  0.7373046875}},
        {"histogram_2d",    {2.22512033291597,  2.22512033291597,  4.07614879956692,  0.79296875}}
        
//        {"mean_2d",         {0.265673476182506, 0.658661581323829, 1.53029574244256,  3.27445219309067}},
//        {"sobel_2d",        {0.218395820507639, 0.509887148867899, 1.08003270414455,  1.41160898683038}},
//        {"laplacian_2d",    {0.269053766577797, 0.741906709445404, 1.43002676010527,  1.71236540749989}},
//        {"fft_2d",          {0.640955962788173, 0.698180086947237, 0.310684776714147, 2./*0.364524768208877*/}},
//        {"dct8x8_2d",       {0.503233693347007, 0.435987033037118, 0.525195542239914, 3./*0.590933568634066*/}},
//        {"hotspot_2d",      {1.29435361772579,  1.72613253937488,  1.71823889626435,  2.188449744359}},
//        {"srad_2d",         {0.700557163661798, 0.609395769414589, 0.466719978570055, 0.614207072612495}},
//        {"dwt_2d",          {3.20018280519802,  3.81926113760053,  3.2507056318647,   3.61916321683164}},
//        {"blackscholes_2d", {1.19454239762865,  2.65023275081232,  4.05394756070832,  4.67577804482401}},
//        {"histogram_2d",    {2.22512033291597,  2.22512033291597,  4.07614879956692,  5.22503008800755}}
    };

    // table["sobel_2d"][1]

};
#endif
