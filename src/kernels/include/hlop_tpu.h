#ifndef __KERNELS_TPU_H__
#define __KERNELS_TPU_H__
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "types.h"
#include "utils.h"
#include "params.h"
#include "gptpu_utils.h"
#include "hlop_base.h"
//#include "CH3_pixel_operation.h"

using namespace cv;

/* edgeTPU kernel class 
TODO: optimize function table searching algorithm.
*/
class HLOPTpu : public HLOPBase{
public:
    HLOPTpu(Params params, void* input, void* output){
        this->params = params;
        this->input = input;
        this->output = output;
        this->kernel_path = get_edgetpu_kernel_path(params.app_name,
                                                    params.get_kernel_size(),
                                                    params.get_kernel_size());
        this->device_handler = new gptpu_utils::EdgeTpuHandler;
        bool verbose = false;
        this->dev_cnt = this->device_handler->list_devices(verbose); 
        for(unsigned int tpuid = 0 ; tpuid < this->dev_cnt ; tpuid++){
            this->device_handler->open_device(tpuid, verbose);
        }
    };

    virtual ~HLOPTpu(){
//        if(this->device_handler != nullptr)
//            delete this->device_handler;
    };
   
    unsigned int get_opened_dev_cnt(){
        return this->dev_cnt;
    }

    /*
        Assign this class instance to 'tpuid' edgetpu in system.
     */
    void set_tpuid(unsigned int tpuid){
        assert(tpuid < this->dev_cnt);
        this->tpuid = tpuid;
    };

    /* input conversion */
    virtual double input_conversion(){
        timing start = clk::now();
        std::string app_name = this->params.app_name;
        this->in_size  = 
            this->params.get_kernel_size() * this->params.get_kernel_size();        
        this->out_size = 
            (app_name == "histogram_2d")?
            256:
            (this->params.get_kernel_size() * this->params.get_kernel_size());        

        // tflite model input array initialization
        if( std::find(this->kernel_table_uint8.begin(),
                      this->kernel_table_uint8.end(),
                      app_name) !=
            this->kernel_table_uint8.end() ){
            this->input_kernel  = reinterpret_cast<uint8_t*>(this->input);
            if(app_name == "histogram_2d"){
                this->output_kernel = (uint8_t*) calloc(256, sizeof(uint8_t));
            }else{
                this->output_kernel = reinterpret_cast<uint8_t*>(this->output);
            }
        }else if( std::find(this->kernel_table_fp.begin(),
                            this->kernel_table_fp.end(),
                            app_name) !=
                  this->kernel_table_fp.end() ){
            if(app_name == "blackscholes_2d"){
                this->in_size  = 
                    3 * this->params.get_kernel_size() * this->params.get_kernel_size();        
            }
            this->input_kernel  = (uint8_t*) malloc(this->in_size * sizeof(uint8_t));
            this->output_kernel = (uint8_t*) calloc(this->out_size, sizeof(uint8_t));
            float* input_array  = reinterpret_cast<float*>(this->input);
            if(app_name == "fft_2d"){
                //this->fft_2d_input_conversion();
                float scale = 255./16.;
#pragma omp parallel for
                for(unsigned int i = 0 ; i < this->in_size ; i++){
                    this->input_kernel[i] = (unsigned)(input_array[i] * scale);
                }
            }else if(app_name == "srad_2d"){
#pragma omp parallel for
                for(unsigned int i = 0 ; i < this->in_size ; i++){
                    this->input_kernel[i] = ((int)(input_array[i] * 255.)) % 256; // float to int conversion
                }
            }else{
#pragma omp parallel for
                for(unsigned int i = 0 ; i < this->in_size ; i++){
                    this->input_kernel[i] = ((int)(input_array[i] /*+ 128*/)) % 256; // float to int conversion
                }
            }
        }else{
            std::cout << __func__ << " [WARN] app: " << app_name 
                      << "is not found in table" << std::endl;
        }
        
        this->model_id = this->device_handler->build_model(this->kernel_path);
        this->device_handler->build_interpreter(rand()%this->dev_cnt, // random
                                                this->model_id);
        this->device_handler->populate_input(this->input_kernel, 
                                             this->in_size, 
                                             this->model_id);

        timing end = clk::now();
        return get_time_ms(end, start);
    }   

    /* output conversion */
    virtual double output_conversion(){
        timing start = clk::now();
        std::string app_name = this->params.app_name;
        float scale;
        uint8_t zero_point;
        this->device_handler->get_raw_output(this->output_kernel, 
                                             this->out_size, 
                                             this->model_id, 
                                             zero_point,
                                             scale);
        //std::cout << __func__ << ": zero_point: " << (unsigned)zero_point << ", scale: " << scale << std::endl;
        if( std::find(this->kernel_table_uint8.begin(),
                      this->kernel_table_uint8.end(),
                      app_name) !=
            this->kernel_table_uint8.end()){
            if(app_name == "histogram_2d"){
                int* int_tmp = reinterpret_cast<int*>(this->output);
                int scale_ = this->params.get_kernel_size() / 256;
                for(unsigned int i = 0 ; i < 256; i++){
                    int_tmp[i] = this->output_kernel[i] * (scale_ * scale_); 
                }
            }else{
                this->output = this->output_kernel; // uint8_t to uint8_t pointer forwarding
            }
        }else if( std::find(this->kernel_table_fp.begin(),
                            this->kernel_table_fp.end(),
                            app_name) !=
                  this->kernel_table_fp.end() ){
            float adj_scale = 1.;
            if(app_name == "fft_2d"){
                adj_scale = scale * 3300.;
            }else if(app_name == "hotspot_2d"){
                adj_scale = scale * 343.76224;
            }else if(app_name == "dct8x8_2d" || app_name == "dwt_2d"){
                adj_scale = scale;
            }else if(app_name == "blackscholes_2d"){
                adj_scale = scale * 29./11.5; // * 29./98.;
            }else if(app_name == "srad_2d"){
                adj_scale = scale ;//* (1./255.);
            }else{
                adj_scale = scale * 255.;
            }
            float* tmp = reinterpret_cast<float*>(this->output);
#pragma omp parallel for
            for(unsigned int i = 0 ; i < this->out_size ; i++){
                tmp[i] = (float)( this->output_kernel[i] - zero_point ) * adj_scale;
            }
        }else{
            std::cout << __func__ << " [WARN] app: " << app_name 
                      << "is not found in table" << std::endl;
        }
        timing end = clk::now();
        return get_time_ms(end, start);
    }

    /*
        TODO: separate out all stage timings of a single run_a_model() call
        will do into input_conversion, actual kernel, output conversion.
        Now measure the e2e time as kernel time as return to align with
        partitioning runtime design. This is a must ToDo as a future performance 
        improvement.
    */
    virtual double run_kernel(unsigned int iter){
        timing start = clk::now();
        this->device_handler->model_invoke(this->model_id, iter);
        timing end = clk::now();
        return get_time_ms(end, start);
    }

private:
    // arrays
    void* input = NULL;
    void* output = NULL;
    uint8_t* input_kernel = NULL;
    uint8_t* output_kernel = NULL;
    unsigned int in_size = 0;
    unsigned int out_size = 0;
    Params params;
    std::string kernel_path;

    // kernel table
    std::vector<std::string> kernel_table_uint8 = {
        "minimal_2d",
        "sobel_2d",
        "mean_2d",
        "laplacian_2d",
        "histogram_2d"
    };
    std::vector<std::string> kernel_table_fp = {
        "fft_2d",
        "dct8x8_2d",
        "blackscholes_2d",
        "hotspot_2d",
        "srad_2d",
        "dwt_2d"
    };
    gptpu_utils::EdgeTpuHandler* device_handler;
    unsigned int dev_cnt = 0;
    unsigned int tpuid = 0;
    unsigned int model_id;
};

#endif
