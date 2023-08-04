#include <time.h>
#include <stdlib.h>
#include "arrays.h"
#include "quality.h"
#include "partition.h"
#include "conversion.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/saliency/saliencySpecializedClasses.hpp>

std::atomic<int> doneProducers(0);
std::atomic<int> doneConsumers(0);

PartitionRuntime::PartitionRuntime(Params params,
                                   std::string mode,
                                   void* input,
                                   void* output){
    params.set_tiling_mode(true);
    this->params = params;
    this->row_cnt = params.get_row_cnt();
    this->col_cnt = params.get_col_cnt();
    this->block_cnt = params.get_block_cnt();
    this->criticality.resize(this->block_cnt);
    assert(this->row_cnt * this->col_cnt == this->block_cnt);
    this->mode = mode;
    this->input = input;
    this->output = output;
    this->generic_kernels = new GenericKernel[this->block_cnt];
    this->dev_sequence = new DeviceType[this->block_cnt];
    this->is_dynamic_block = new bool[this->block_cnt]; 
    this->is_dynamic_device = new bool[this->dev_type_cnt+1]; // enum is 1-index. 
    // For rand_p partition mode
    srand(time(NULL));
};

PartitionRuntime::~PartitionRuntime(){
    for(unsigned int i = 0 ; i < this->block_cnt ; i++){
        delete this->generic_kernels[i].kernel_base;
    }   
    delete this->generic_kernels;
    delete this->dev_sequence;
    delete this->is_dynamic_block;
    delete this->is_dynamic_device;
    this->input_pars.clear();
    this->output_pars.clear();
    this->sampling_qualities.clear();
    this->criticality.clear();
}

bool sortByVal(const std::pair<int , float> &a, const std::pair<int, float> &b){
    return a.second < b.second;
}

/*
 This is the main function to determine criticality of each tiling block based on
 sampling quality.
 */
void PartitionRuntime::criticality_kernel(Params params, 
                                          std::vector<std::pair<int, float>>& order, 
                                          float criticality_ratio,
                                          std::string mode/*threshold, topK*/){

    /*
        Current design: mark (no more than) one third of the worst blocks 
            (error_rate worst) to be critical.
        TODO: design the criticality decision 
     */
    //std::cout << __func__ << ": criticality method: " << mode << std::endl;
    if(mode == "c-limits"){
        float threshold = 256.;
        for(auto p: order){
            //std::cout << __func__ << ": p.second: " << p.second << std::endl;
            this->criticality[p.first] = (p.second >= threshold)?true: false;
        }
    }else if(mode == "topK"){
        int window_size = 256;
        float c = params.get_criticality_ratio();
        int K = int(window_size*c);
        int window_cnt = 256./256.; //((int)order.size() / window_size) + (order.size()%window_size == 0)?0:1;
        for(int i = 0 ; i < window_cnt ; i++){
            std::vector<std::pair<int, float>> window;
            for(int j = 0 ; j < window_size ; j++){
                int idx = i*window_cnt+j;
                idx = (idx > order.size())?order.size():idx;
                window.push_back(order[idx]);
            } 
            sort(window.begin(), window.end(), sortByVal);
            for(int j = 0 ; j < window_size ; j++){
                this->criticality[window[j].first] = (j <= (window_size-K))?false:true;
                
            }
            window.resize(0);
        }
    //sort(order.begin(), order.end(), sortByVal);
    }else{
        std::cout << __func__ << ": unknown criticality mode(not limits or topK): "
                  << mode << std::endl;
        exit(0);
    }
/*
    std::cout << __func__ << ": criticality ratio: " << criticality_ratio << ", order size: " << order.size() << std::endl;
    int threshold = ceil(order.size() * (1. - criticality_ratio));
    int cnt = 0;
    std::cout << __func__ << ": (idx, criticality)" << std::endl;;
    for(auto p: order){
        this->criticality[p.first] = (cnt < threshold)?false:true;
        cnt++;
        std::cout << "idx: (" << p.first/16 << ", " << p.first%16 << "), metirc: " << p.second << std::endl;
    }
*/
    // show criticality
    //std::cout << __func__ << ": criticality tiling:" << std::endl;
    int cnt = 0;
    for(unsigned int i = 0 ; i < params.get_row_cnt() ; i++){
        for(unsigned int j = 0 ; j < params.get_col_cnt() ; j++){
            unsigned int idx = i * params.get_col_cnt() + j;
            cnt += (this->criticality[idx] == true)?1:0;
            //std::cout << ((this->criticality[idx] == true)?"g":"t") << " ";
        }
        //std::cout << std::endl;
    }
    //std::cout << __func__ << ": critical ratio: " << cnt << "/" << params.get_row_cnt() * params.get_col_cnt()
    //          << "(" << (float)cnt/(params.get_row_cnt()*params.get_col_cnt())  << "): waitting..." << std::endl;
    //std::cout << std::endl;
}

double PartitionRuntime::run_sampling(SamplingMode mode){
/*
    Sampling policies:
    
    0. Oracle: do a acutal full scale run to know the quality rank
    1. fixed number of pixel samplings (input stats):
        1-1: one pixel per sub-tiling block
        1-2: N stride pixels per sub-tiling block, N is constant
            within N pixels, find minmax to represent full tiling block's dist.
        1-3: N random pixels per sub-tiling block, N is constant
            within N pixels, find minmax to represent full tiling block's dist.
    2. fixed percentage of pixel samplings (actual run sampling):
        2-1. sub-tiling block downsampling
 */
    
    this->params.set_sampling_mode(mode);
    //std::cout << __func__ << ": start sampling run, mode: " 
    //          << this->params.get_sampling_mode() 
    //          << ", downsampling rate: "
    //          << this->params.get_downsampling_rate() << std::endl;
    /* Downsampling tiling blocks and assign them to edgetpu. */
    std::vector<void*> input_sampling_pars;
    std::vector<void*> gpu_output_sampling_pars;
    std::vector<void*> tpu_output_sampling_pars;
    
    array_partition_downsampling(this->params,
                                 false,
                                 this->input_pars,
                                 input_sampling_pars);
    
    array_partition_downsampling(this->params,
                                 true, // skip_init
                                 this->output_pars,
                                 gpu_output_sampling_pars);
    
    array_partition_downsampling(this->params,
                                 true, // skip_init
                                 this->output_pars,
                                 tpu_output_sampling_pars);

    double sampling_overhead = 0.0;

    /* run downsampled tiling blocks on edgetpu and get quality result. */
    
    for(unsigned int i = 0 ; i < this->row_cnt ; i++){
        for(unsigned int j = 0; j < this->col_cnt ; j++){
            unsigned int idx = i*this->col_cnt+j;
            Params params = this->params;
            params.block_size = this->params.block_size * params.get_downsampling_rate();

            // cpu part
             HLOPGpu* gpu_kernel_ptr = new HLOPGpu(params,
                                                       input_sampling_pars[idx],
                                                       gpu_output_sampling_pars[idx]);
            sampling_overhead += gpu_kernel_ptr->input_conversion();
            sampling_overhead += gpu_kernel_ptr->run_kernel(1);
            sampling_overhead += gpu_kernel_ptr->output_conversion();

            // tpu part
            HLOPTpu* tpu_kernel_ptr = new HLOPTpu(params,
                                                      input_sampling_pars[idx],
                                                      tpu_output_sampling_pars[idx]);
            sampling_overhead += tpu_kernel_ptr->input_conversion();
            sampling_overhead += tpu_kernel_ptr->run_kernel(1);
            sampling_overhead += tpu_kernel_ptr->output_conversion();

            params.problem_size = params.block_size;
            
            UnifyType* unify_input_type = 
                new UnifyType(params, input_sampling_pars[idx]);          
            UnifyType* unify_gpu_output_type =
                new UnifyType(params, gpu_output_sampling_pars[idx]);
            UnifyType* unify_tpu_output_type =
                new UnifyType(params, tpu_output_sampling_pars[idx]);
                
//            Quality* quality = new Quality(params.block_size, // m
//                                           params.block_size, // n
//                                           params.block_size, // ldn
//                                           params.block_size,
//                                           params.block_size,
//                                           unify_input_type->float_array,
//                                           unify_tpu_output_type->float_array,
//                                           unify_cpu_output_type->float_array);
            std::vector<bool> dummy(1, true);
            std::vector<int> dummy2(1, 2);
            this->sampling_qualities.push_back(Quality(params.app_name,
                                                       params.block_size,
                                                       params.block_size,
                                                       params.block_size,
                                                       params.block_size,
                                                       params.block_size,
                                                       unify_input_type->float_array,
                                                       unify_tpu_output_type->float_array,
                                                       unify_gpu_output_type->float_array,
                                                       //dummy,
                                                       dummy2));
//            std::cout << __func__ << ": block[" << i << ", " << j << "]: "
//                      << "rmse: " << quality->rmse()
//                      << ", error_rate: " << quality->error_rate()
//                      << ", error_percentage: " << quality->error_percentage()
//                      << ", ssim: " << quality->ssim()
//                      << ", pnsr: " << quality->pnsr() << std::endl;
        }
    }


    //std::cout << __func__ << ": sampling timing overhead: " << sampling_overhead << " (ms)" << std::endl;
    /* criticality policy to determine which tiling block(s) are critical. */
    std::vector<std::pair<int, float>> order;

    for(unsigned int i = 0 ; i < this->sampling_qualities.size() ; i++){
//        std::cout << __func__ << ": (" << i
//                  << ", rmse: " << this->sampling_qualities[i].rmse()
//                  << ", rmse %: " << this->sampling_qualities[i].rmse_percentage()
//                  << ", error rate: " << this->sampling_qualities[i].error_rate()
//                  << ", error %: " << this->sampling_qualities[i].error_percentage()
//                  << ", ssim: " << this->sampling_qualities[i].ssim()
//                  << ", pnsr: " << this->sampling_qualities[i].pnsr() 
//                  << std::endl;
        // objective or criticality ordering
        //order.push_back(std::make_pair(i, (-1)*this->sampling_qualities[i].ssim()));
        if(params.app_name == "mean_2d" || params.app_name == "sobel_2d"){
            order.push_back(std::make_pair(i, this->sampling_qualities[i].rate())); // to ordering error_rate
        }else if(params.app_name == "laplacian_2d"){
            order.push_back(std::make_pair(i, -1*this->sampling_qualities[i].rate())); // to ordering error_rate
        }else{
            order.push_back(std::make_pair(i, this->sampling_qualities[i].rate())); // to ordering error_rate
        }
        /* good for sobel_2d */
        //order.push_back(std::make_pair(i, this->sampling_qualities[i].rmse()));
        
        /* testing for laplacian_2d */
        //order.push_back(std::make_pair(i, this->sampling_qualities[i].rmse()));
    }
    sort(order.begin(), order.end(), sortByVal);

    //std::cout << __func__ << ": oracle s.t. rmse." << std::endl;
    
    //std::cout << __func__ << ": oracle c ratio: " << params.get_criticality_ratio() << std::endl;
    std::string critical_mode = "topK";
    this->criticality_kernel(params, order, params.get_criticality_ratio()/*for stable oracle*/, critical_mode); 

    // If a block is critical, then it must be static and assigned to GPU.
    // And for non-critical blocks, it is dynamic and upto runtime to determine device type to run.
    // In this way, work stealing is used during runtime.
    
    return sampling_overhead;
}

double sdev(std::vector<float> v){
    double sum = 0.;
    double square_sum = 0.;
    auto const count = static_cast<float>(v.size());
    double mean;
    for(auto p: v){
        sum += p;
    }
    mean = sum / count;
    for(auto p: v){
        square_sum += pow(p - mean, 2);
    }
    return pow((double)(square_sum / (double)(count)), 0.5);
}

double PartitionRuntime::run_input_stats_probing(std::string mode, unsigned int num_pixels){
    timing s = clk::now();
    std::vector<float> samples_max(this->params.get_block_cnt(), FLT_MIN);
    std::vector<float> samples_min(this->params.get_block_cnt(), FLT_MAX);
    std::vector<float> samples_range(this->params.get_block_cnt(), 0.);
    std::vector<float> samples_sdev(this->params.get_block_cnt(), 0.);
    std::vector<std::vector<float>> samples_pixels(this->params.get_block_cnt());
    int total_size = this->params.get_kernel_size() * this->params.get_kernel_size();
    unsigned int offset;
    float tmp;
    for(unsigned int i = 0 ; i < this->params.get_block_cnt() ; i++){
        for(unsigned int idx = 0 ; idx < num_pixels ; idx++){
            if(mode == "c-tu" || mode == "c-ku"){
                offset = rand()%total_size;
            }else if(mode == "c-ts" || mode == "c-ks"){ // stride from top-left corner
                unsigned int stride = 32; // just a default
                offset = (idx * stride) % total_size; // flat circular stridding to avoid segflt
            }else{
                std::cout << __func__ << ": input stats sampling mode: " 
                            << mode << " is not supported yet." << std::endl;
                exit(0);
            }
            if(std::find(this->params.uint8_t_type_app.begin(),
                        this->params.uint8_t_type_app.end(),
                        this->params.app_name) !=
                    this->params.uint8_t_type_app.end()){
                uint8_t* ptr = reinterpret_cast<uint8_t*>(this->input_pars[i]);
                tmp = ptr[offset]; // uchar to float conversion
            }else{
                float* ptr = reinterpret_cast<float*>(this->input_pars[i]);
                tmp = ptr[offset];
            }
            samples_pixels[i].push_back(tmp);
            samples_max[i] = (tmp > samples_max[i])?tmp:samples_max[i];
            samples_min[i] = (tmp < samples_min[i])?tmp:samples_min[i];
        }
    }
    /*
    for(unsigned int i = 0 ; i < this->params.get_block_cnt() ; i++){
        samples_sdev[i] = sdev(samples_pixels[i]);
    }*/

    std::vector<std::pair<int, float>> order;
    for(unsigned int i = 0 ; i < this->params.get_block_cnt() ; i++){
        assert(samples_max[i] >= samples_min[i]);
        samples_range[i] = samples_max[i] - samples_min[i];
        order.push_back(std::make_pair(i, samples_range[i]));
    }
    //sort(order.begin(), order.end(), sortByVal);
   
    //std::cout << __func__ << ": criticlaity ratio: " << this->params.get_criticality_ratio() << std::endl;

    std::string critical_mode = "c-limits";
    this->criticality_kernel(this->params, order, this->params.get_criticality_ratio(), critical_mode); 

    timing e = clk::now();
    return get_time_ms(e, s); 
}

double PartitionRuntime::run_input_homo_probing(float one_dim_ratio){
    timing s = clk::now();
    std::vector<float> samples_homo_cnt(this->params.get_block_cnt(), 0.);
    std::vector<float> samples_mean(this->params.get_block_cnt(), 0.);
    std::vector<float> samples_sdev(this->params.get_block_cnt(), 0.);
    std::vector<std::vector<float>> samples_pixels(this->params.get_block_cnt());
    int w = this->params.get_kernel_size();
    int h = this->params.get_kernel_size();
    int s_w = one_dim_ratio * w; // downsized w
    int s_h = one_dim_ratio * h; // downsized w
    int total_size = w * h;
    unsigned int offset;
    float tmp;
    Quality* q = new Quality();
    for(unsigned int i = 0 ; i < this->params.get_block_cnt() ; i++){
        // 1. get the downsampling canary
        if(std::find(this->params.uint8_t_type_app.begin(),
                    this->params.uint8_t_type_app.end(),
                    this->params.app_name) !=
                this->params.uint8_t_type_app.end()){
            uint8_t* ptr = reinterpret_cast<uint8_t*>(this->input_pars[i]);
            cv::Mat mat(w, h, CV_8U), sample_mat(s_w, s_h, CV_8U);
            array2mat(mat, ptr, w, h);
            cv::resize(mat, 
                       sample_mat,
                       cv::Size(s_w, s_h), 0, 0,
                       INTER_NEAREST);
            uint8_t* sample_array = (uint8_t*) malloc(s_w * s_h * sizeof(uint8_t));
            mat2array(sample_mat, sample_array);
            samples_sdev[i] = q->static_sdev(sample_array, s_w * s_h);
            samples_mean[i] = q->static_mean(sample_array, s_w * s_h);
//#pragma omp parallel for
            for(int i_idx = 0 ; i_idx < s_w ; i_idx++){
                for(int j_idx = 0 ; j_idx < s_h ; j_idx++){
                    samples_homo_cnt[i] += (fabs(sample_array[i_idx * s_h + j_idx] - samples_mean[i]) < samples_sdev[i])?1:0;
                }
            }
        }else{
            float* ptr = reinterpret_cast<float*>(this->input_pars[i]);
            cv::Mat mat(w, h, CV_32F), sample_mat(s_w, s_h, CV_32F);
            array2mat(mat, ptr, w, h);
            cv::resize(mat, 
                       sample_mat,
                       cv::Size(s_w, s_h), 0, 0,
                       INTER_NEAREST);
            float* sample_array = (float*) malloc(s_w * s_h * sizeof(float));
            mat2array(sample_mat, sample_array);
            samples_sdev[i] = q->static_sdev(sample_array, s_w * s_h);
            samples_mean[i] = q->static_mean(sample_array, s_w * s_h);
#pragma omp parallel for
            for(int i_idx = 0 ; i_idx < s_w ; i_idx++){
                for(int j_idx = 0 ; j_idx < s_h ; j_idx++){
                    samples_homo_cnt[i] += (fabs(sample_array[i_idx * s_h + j_idx] - samples_mean[i]) < samples_sdev[i])?1:0;
                }
            }
        }
    }

    std::vector<std::pair<int, float>> order;
    for(unsigned int i = 0 ; i < this->params.get_block_cnt() ; i++){
        std::cout << __func__ << ": homo cnt[" << i << "]: " << samples_homo_cnt[i] << ", block cnt:" << this->params.get_block_cnt() << std::endl;
        order.push_back(std::make_pair(i, samples_sdev[i]));
        //order.push_back(std::make_pair(i, samples_homo_cnt[i]));
    }
    //sort(order.begin(), order.end(), sortByVal);
   
    std::cout << __func__ << ": criticlaity ratio: " << this->params.get_criticality_ratio() << std::endl;
    std::string critical_mode = "c-limits";
    this->criticality_kernel(this->params, order, this->params.get_criticality_ratio(), critical_mode); 

    timing e = clk::now();
    return get_time_ms(e, s); 
}

double PartitionRuntime::set_criticality_by_saliency(Params params, void** array){
    timing s = clk::now();
    cv::Mat mat;
    cv::Mat saliency_map, binary_map;
    auto saliency = cv::saliency::StaticSaliencySpectralResidual();
        
    if(std::find(this->params.uint8_t_type_app.begin(),
                 this->params.uint8_t_type_app.end(),
                 this->params.app_name) !=
       this->params.uint8_t_type_app.end()){
        array2mat(mat, (uint8_t*)*array, params.problem_size, params.problem_size);
    }else{
        array2mat(mat, (float*)*array, params.problem_size, params.problem_size);
    } 

    std::cout << __func__ << ": calc saliency..." << std::endl;

    assert(saliency.computeSaliency(mat, saliency_map));
    assert(saliency.computeBinaryMap(saliency_map, binary_map));

    unsigned long long int total_saliency_cnt = 0;
    unsigned long long int total_pixel_cnt = 0;
    std::vector<std::pair<int, float>> saliency_ratio;

    for(unsigned int i_idx = 0 ; i_idx < params.get_row_cnt() ; i_idx++){
        for(unsigned int j_idx = 0 ; j_idx < params.get_col_cnt() ; j_idx++){
            unsigned long long int total_cnt = 0;
            unsigned long long int saliency_cnt = 0;
            unsigned int i_start = i_idx * params.get_kernel_size();
            unsigned int j_start = j_idx * params.get_kernel_size();
            int idx = i_idx*params.get_col_cnt()+j_idx;
            for(unsigned int i = i_start ; i < i_start+params.get_kernel_size() ; i++){
                for(unsigned int j = j_start ; j < j_start+params.get_kernel_size() ; j++){
                    saliency_cnt += ((uint8_t)binary_map.at<uint8_t>(i, j))?1:0;
                    total_cnt++;
                }
            }
            total_saliency_cnt += saliency_cnt;
            total_pixel_cnt += total_cnt;
//            std::cout << __func__ << ": " << i_idx << ", " << j_idx 
//                      << ": saliency_cnt: " << saliency_cnt 
//                      << ", total: " << total_cnt 
//                      << ", rate: " 
//                      << (float)saliency_cnt/total_cnt << std::endl;
            saliency_ratio.push_back(std::make_pair(idx, (float)saliency_cnt/total_cnt));
            this->criticality[idx] = (((float)saliency_cnt / total_cnt) > 0.)?true:false;
        }
    }
    float saliency_rate = (float)total_saliency_cnt / total_pixel_cnt;
    std::cout << __func__ 
              << ": total saliency rate: " 
              << saliency_rate 
              << ", total non-saliency rate: "
              << 1. - saliency_rate
              << std::endl;
    
    sort(saliency_ratio.begin(), saliency_ratio.end(), sortByVal);
    //this->criticality_kernel(params, saliency_ratio, params.get_criticality_ratio()); 

    std::cout << __func__ << ": criticality tiling:" << std::endl;
    for(unsigned int i = 0 ; i < params.get_row_cnt() ; i++){
        for(unsigned int j = 0 ; j < params.get_col_cnt() ; j++){
            unsigned int idx = i * params.get_col_cnt() + j;
            std::cout << ((this->criticality[idx] == true)?"g":"t") << " ";
        }
        std::cout << std::endl;
    }
    timing e = clk::now();
    return get_time_ms(e, s);
}

double PartitionRuntime::prepare_partitions(){
    double ret = 0.0;

    // allocate input partitions and initialization
    array_partition_initialization(this->params,
                                   false,
                                   &(this->input),
                                   this->input_pars);

    // allocate output partitions
    array_partition_initialization(this->params,
                                   true, // skip_init
                                   &(this->output),
                                   this->output_pars);

/* sampling section if enabled */
    if(this->is_criticality_mode() || this->get_partition_mode() == "s"){
        auto p_mode = this->get_partition_mode();
        //std::cout << __func__ << ": mode: " << p_mode << std::endl;

        // TODO: set back after testing across various criticality ratio
        this->params.set_criticality_ratio();
        //std::cout << __func__ << ": criticality ratio: " << this->params.get_criticality_ratio() << std::endl;
        // involves actual run types
        if(p_mode == "c-saliency"){
            ret += this->set_criticality_by_saliency(this->params, &(this->input));
        }else if(p_mode == "c-oracle"){ // full scale run test
            // To determine critical or not on each tiling block by saliency detection
            //this->params.set_criticality_ratio(1./3.);
            SamplingMode mode = init_crop;
            this->params.set_downsampling_rate(1.);
            ret += this->run_sampling(mode);
        }else if(p_mode == "c"){ // use default downsampling rate
            SamplingMode mode = center_crop;
            //this->params.set_downsampling_rate(1./8.);
            ret += this->run_sampling(mode);
        }else if(p_mode == "c-ts" || 
                 p_mode == "c-tu" ||
                 p_mode == "c-ks" ||
                 p_mode == "c-ku"){
            int num_pixels = this->params.get_num_sample_pixels();;
            //std::cout << __func__ << ": num sample pixels: " << num_pixels << std::endl;
            ret += this->run_input_stats_probing(p_mode, num_pixels); 
        }else if(p_mode == "c-tr" ||
                 p_mode == "c-kr"){
            float one_dim_ratio = 1/16.;
            ret += this->run_input_homo_probing(one_dim_ratio); 
        }else{
            std::cout << __func__ 
                      << ": unknown partition mode: " << p_mode << std::endl;
            exit(0);
        }
    }
/* end sampling section */
        
    this->setup_dynamic_devices();

    /* This is the latest moment to determine if each tiling block and device is 
       dynamic or static. */
    this->setup_dynamic_blocks();
    
    // simulating oracle
    //SamplingMode mode = init_crop;
    //this->params.set_downsampling_rate(1.);
    //this->run_sampling(mode); 
    
    // assign partitions to corresponding type of kernel handler if is static.
    for(unsigned int i = 0 ; i < this->row_cnt ; i++){
        for(unsigned int j = 0; j < this->col_cnt ; j++){
            unsigned int idx = i*this->col_cnt+j;
            //std::cout << __func__ << ": is_dynamic_block[" << i << "]: " << this->is_dynamic_block[idx] << std::endl;
            if( !this->is_dynamic_block[idx] ){
                auto device_type = this->mix_policy(idx);
                this->create_kernel_by_type(idx, device_type);
                ret += this->generic_kernels[idx].kernel_base->input_conversion();
            }
        }
    }
    return ret;
}

void* PartitionRuntime::RunDeviceThread(void *my_args){
    // getting argument(s)
    struct thread_data *args = (struct thread_data*) my_args;
    auto p_run_ptr = args->p_run_ptr; // pointer of 'this'
    GenericKernel* generic_kernels = args->generic_kernels;
    unsigned int block_cnt = args->block_cnt;
    unsigned int iter = args->iter;
    double kernel_ms = args->kernel_ms;
    DeviceType device_type = args->device_type;
    
    kernel_ms = 0.0;
    
    // To consume any tiling block that is assigned to this device statically.
    for(unsigned int i = 0 ; i < block_cnt ; i++){
        /* Check if the device type this kernel was assigned to is the same as
           the type this consumer thread is representing. 
         */
        if(p_run_ptr->is_dynamic_block[i] == false &&
            generic_kernels[i].device_type == device_type){
            kernel_ms += generic_kernels[i].kernel_base->run_kernel(iter);
        }
    }
    
    timing start = clk::now();
    // device as dynamic consumer
    if(p_run_ptr->is_dynamic_device[device_type]){
        struct node_data curr_node;
        bool itemsLeft;
        do{
            itemsLeft = doneProducers.load(std::memory_order_acquire) != 1;
            while(p_run_ptr->q.try_dequeue(curr_node)){
                itemsLeft = true;
            /*  Start to consume one tiling block.
                Current implementation has to include input conversion overhead 
                since device type is not determined until now.
            */
            unsigned int block_id = curr_node.block_id;
            p_run_ptr->create_kernel_by_type(block_id, device_type);
            p_run_ptr->dev_sequence[block_id] = device_type;
            curr_node.generic_kernel->kernel_base->input_conversion();
            kernel_ms += 
                curr_node.generic_kernel->kernel_base->run_kernel(curr_node.iter);
            }
        }while(itemsLeft || 
                doneConsumers.fetch_add(1, std::memory_order_acq_rel) + 1 == 
                (int)p_run_ptr->dev_type_cnt);
    }
    timing end = clk::now();
    double e2e_kernel_ms = get_time_ms(end, start);
    //std::cout << __func__ << ": e2e kernel time: " 
    //                      << e2e_kernel_ms << 
    //                      " (ms), HW busy time: " 
    //                      << kernel_ms << " (ms), comm. overhead: "
    //                      << e2e_kernel_ms - kernel_ms << " (ms)." << std::endl;

    args->kernel_ms = kernel_ms;
    pthread_exit(NULL);
}

double PartitionRuntime::run_partitions(){
    timing start = clk::now();
    /*
       Dynamic producer of SPMC scheduling.
       Any dynamic tiling block that is left un-assigned to any device during
       static assignment stage now will be push into SPMC FIFO queue for 
       dynamic scheduling.
    */
    for(unsigned int i = 0 ; i < this->block_cnt ; i++){
        //std::cout << __func__ << ": is_dynamic_block[" << i << "]: " << this->is_dynamic_block[i] << std::endl;
        if(this->is_dynamic_block[i]){
            struct node_data curr_node;
            curr_node.generic_kernel = &(this->generic_kernels[i]);
            curr_node.params = this->params;
            curr_node.block_id = i;
            curr_node.iter = this->params.iter;
            this->q.enqueue(curr_node);
        }
    }
    doneProducers.fetch_add(1, std::memory_order_release);

    //create pthreads for each device as runtime threading
    pthread_t threads[this->dev_type_cnt];
    struct thread_data td[this->dev_type_cnt];

    // CPU thread
    td[0].device_type = cpu;
    
    // GPU thread
    td[1].device_type = gpu;

    // edgeTPU thread
    td[2].device_type = tpu;
    
    // create device threads
    for(unsigned int i = 0 ; i < this->dev_type_cnt ; i++){
        td[i].p_run_ptr = this;
        td[i].generic_kernels = this->generic_kernels; 
        td[i].block_cnt = this->block_cnt;
        td[i].iter = this->params.iter;
        pthread_create(&threads[i], NULL, this->RunDeviceThread, (void *)&td[i]);
    }

    // wait for join
    for(unsigned int i = 0 ; i < this->dev_type_cnt ; i++){
        pthread_join(threads[i], NULL);
    }
    timing end = clk::now();
    //std::cout << __func__ << ": CPU thread latency: " << td[0].kernel_ms << " (ms)" << std::endl;
    //std::cout << __func__ << ": GPU thread latency: " << td[1].kernel_ms << " (ms)" << std::endl;
    //std::cout << __func__ << ": TPU thread latency: " << td[2].kernel_ms << " (ms)" << std::endl;
    double e2e_kernel_ms = get_time_ms(end, start);
    //std::cout << __func__ << ": e2e kernel time: " << e2e_kernel_ms << " (ms) (pthread overhead included)" << std::endl;
    return e2e_kernel_ms;
}

double PartitionRuntime::transform_output(){
    double ret = 0.0;
    for(unsigned int  i = 0 ; i < this->block_cnt ; i++){
        ret +=  this->generic_kernels[i].kernel_base->output_conversion();
    }  
    output_array_partition_gathering(this->params,
                                     &(this->output),
                                     this->output_pars);
    return ret;
}

void PartitionRuntime::create_kernel_by_type(unsigned int i/*block_id*/, 
                                             DeviceType device_type){
    if(this->generic_kernels[i].kernel_base != NULL){
        std::cout << "[WARN] " << __func__ << ": generic_kenrels[" << i 
                  << "] has been instanciated as type " 
                  << this->generic_kernels[i].device_type 
                  << ", and now type " << device_type 
                  << " is wanted. Skip creating." << std::endl;
    }else{
        if(device_type == cpu){
            this->generic_kernels[i].kernel_base =
                new HLOPCpu(this->params,
                            this->input_pars[i],
                            this->output_pars[i]);
            this->generic_kernels[i].device_type = cpu;
        }else if(device_type == gpu){
            this->generic_kernels[i].kernel_base =
                new HLOPGpu(this->params,
                              this->input_pars[i],
                              this->output_pars[i]);
            this->generic_kernels[i].device_type = gpu;
        }else if(device_type == tpu){
            this->generic_kernels[i].kernel_base =
                new HLOPTpu(this->params,
                              this->input_pars[i],
                              this->output_pars[i]);
            this->generic_kernels[i].device_type = tpu;
        }else{
           std::cout << __func__ << ": undefined device type: "
                     << device_type 
                     << " on block id: " << i
                     << ", program exits."
                     << std::endl;
           exit(0);
        }
    }
}

DeviceType PartitionRuntime::mix_policy(unsigned i
        /*index of a tiling task, no larger than this->block_cnt*/){
    DeviceType ret = undefine;
    if(this->mode == "c_p"){ // all partitions on cpu
        ret = cpu;
    }else if(this->mode == "g_p"){ // all partitions on gpu
        ret = gpu;
    }else if(this->mode == "t_p"){ // all partitions on tpu
        ret = tpu;
    }else if(this->mode == "cgt_s"){ // sequentially choose a device between cpu, gpu and tpu
        int idx = i%3;
        ret = (idx == 0)?cpu:((idx == 1)?gpu:tpu);
    }else if(this->mode == "cg_s"){ // sequentially choose between cpu and gpu
        ret = (i%2 == 0)?cpu:gpu;
    }else if(this->mode == "gt_s"){ // sequentially choose between gpu and tpu
        ret = (i%2 == 0)?gpu:tpu;
        
        // simulating naive work balancing
        //this->params.set_criticality_ratio();
        //ret = (((double)rand()/RAND_MAX) <= this->params.get_criticality_ratio())?gpu:tpu;
        //std::cout << __func__ << ": c[" << i << ": " << this->criticality[i] << std::endl;
        ret = ((this->criticality[i] == true)?gpu:tpu);

        //ret = (this->criticality[i] == true)?gpu:tpu;
    }else if(this->mode == "ct_s"){ // sequentially choose between cpu and tpu
        ret = (i%2 == 0)?cpu:tpu;
    }else if(this->mode == "cgt_b" ||
             this->mode == "cg_b" ||
             this->mode == "gt_b" ||
             this->mode == "ct_b"){
        /*
           For work-balancing type of modes, device assignment of each tiling 
           block is dynamic (determined by SPMC at runtime). No need to 
           pre-determine here so do nothing.
         */
    }else if(this->mode == "gt_c" ||
             this->mode == "gt_c-saliency" ||
             this->mode == "gt_c-oracle" ||
             this->mode == "gt_c-ts" ||
             this->mode == "gt_c-tu" ||
             this->mode == "gt_c-tr" ||
             this->mode == "gt_c-ks" ||
             this->mode == "gt_c-ku" ||
             this->mode == "gt_c-kr" ){ // criticality mode on GPU/TPU mixing
        ret = (this->criticality[i] == true)?gpu:undefine; // non-critical blocks are dynamic
    }else{
        std::cout << __func__ << ": undefined partition mode: "
                  << this->mode << ", program exits."
                  << std::endl;
        exit(0);
    }   
    this->dev_sequence[i] = ret;
    return ret;
}

void PartitionRuntime::show_device_sequence(){
    std::cout << __func__ << ": (in [i, j] indexing)" << std::endl;
    for(unsigned int  i = 0 ; i < this->row_cnt ; i++){
        for(unsigned int j = 0 ; j < this->col_cnt ; j++){
            unsigned int idx = i*this->col_cnt +j;
            int tmp = this->dev_sequence[idx];
            if(tmp == cpu){
                std::cout << "c";
            }else if(tmp == gpu){
                std::cout << "g";
            }else if(tmp == tpu){
                std::cout << "t";
            }
            std::cout << " ";
        }
        std::cout << std::endl;
    }   
}

std::vector<DeviceType> PartitionRuntime::get_device_sequence(){
    std::vector<DeviceType> ret;
    for(unsigned int i = 0 ; i < this->row_cnt ; i++){
        for(unsigned int j = 0 ; j < this->col_cnt ; j++){
            unsigned int idx = i*this->col_cnt+j;
            ret.push_back(this->dev_sequence[idx]);
        }
    }
    return ret;
}

bool PartitionRuntime::is_criticality_mode(){
    bool ret = false;
    unsigned int delimiter_loc = this->mode.find("_");
    if(delimiter_loc != std::string::npos && 
        this->mode.length() > delimiter_loc &&
        this->mode.substr(delimiter_loc+1, 1) == "c"){
        ret = true;
    }
    return ret;
}

std::string PartitionRuntime::get_partition_mode(){
    std::string ret;
    unsigned int delimiter_loc = this->mode.find("_");
    if(delimiter_loc != std::string::npos && 
        this->mode.length() > delimiter_loc){
        ret = this->mode.substr(delimiter_loc+1);
    }else{
        std::cout << __func__ 
                  << ": no partition mode found, exit." << std::endl;
        exit(0);
    }
    return ret;
}

/* Setup default dynamic flag based on this->mode */
void PartitionRuntime::setup_dynamic_blocks(){
    unsigned int delimiter_loc = this->mode.find("_");
    
    if(is_criticality_mode()){
        assert(this->criticality.size() == this->block_cnt);    
        for(unsigned int i = 0 ; i < this->block_cnt ; i++){
            this->is_dynamic_block[i] = (this->criticality[i] == true)?false:true;
        }
    }else if(delimiter_loc != std::string::npos && 
             this->mode.length() > delimiter_loc &&
             this->mode.substr(delimiter_loc+1, 1) == "b"){
        // default as dynamic
        for(unsigned int i = 0 ; i < this->block_cnt ; i++){
            this->is_dynamic_block[i] = true;
        }
    
    }else{ // all other non criticality aware non-sampling policies
        // default as static
        for(unsigned int i = 0 ; i < this->block_cnt ; i++){
            this->is_dynamic_block[i] = false;
        }
    }
}

/* Setup default dynamic flag based on this->mode */
void PartitionRuntime::setup_dynamic_devices(){
    unsigned int delimiter_loc = this->mode.find("_");
    
    // default as static
    for(unsigned int i = 0 ; i < this->dev_type_cnt+1/*enum is 1-index*/ ; i++){
        this->is_dynamic_device[i] = false;
    }

    if((delimiter_loc != std::string::npos && 
        this->mode.length() > delimiter_loc &&
        this->mode.substr(delimiter_loc+1, 1) == "b") ||
        this->is_criticality_mode()){
        
        // switch each device to dynamic if detected.
        std::string sub_mode = this->mode.substr(0, delimiter_loc);
        if(sub_mode.find("c") != std::string::npos){ // found cpu type
            this->is_dynamic_device[cpu] = true;
        }
        if(sub_mode.find("g") != std::string::npos){ // found gpu type
            this->is_dynamic_device[gpu] = true;
        }
        if(sub_mode.find("t") != std::string::npos){ // found tpu type
            this->is_dynamic_device[tpu] = true;
        }
    }// else: no partition mode(s). all static
}

