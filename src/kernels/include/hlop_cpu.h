#ifndef __KERNELS_CPU_H__
#define __KERNELS_CPU_H__
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "types.h"
#include "utils.h"
#include "params.h"
#include "hlop_base.h"

using namespace cv;

/* CPU kernel class 
TODO: optimize function table searching algorithm.
*/
class HLOPCpu : public HLOPBase{
public:
    HLOPCpu(Params params, void* input, void* output){
        this->params = params;
        this->input_array_type.ptr = input;
        this->output_array_type.ptr = output;
    };

    virtual ~HLOPCpu(){};
    
    /* input conversion - search over func_tables to do correct input conversion */
    virtual double input_conversion(){
        timing start = clk::now();
        std::string app_name = this->params.app_name;
        if(if_kernel_in_table(this->func_table_cv, app_name)){
            uint8_t* input_array  = 
                reinterpret_cast<uint8_t*>(this->input_array_type.ptr);
            array2mat(this->input_array_type.mat, 
                      input_array, 
                      this->params.get_kernel_size(), 
                      this->params.get_kernel_size());
        }else if(if_kernel_in_table(this->func_table_fp, app_name)){
            float* input_array  = 
                reinterpret_cast<float*>(this->input_array_type.ptr);
            float* output_array = 
                reinterpret_cast<float*>(this->output_array_type.ptr);
            if(app_name == "dct8x8_2d"){
                this->input_array_type.fp 
                    = (float*) malloc(this->params.get_kernel_size() * 
                                      this->params.get_kernel_size() * 
                                      sizeof(float));
                int StrideF = ((int)ceil(this->params.get_kernel_size()/16.0f))*16;
                // ***** float shifting *****
                //AddFloatPlane(-128.0f, input, StrideF, ImgSize);
                assert((unsigned int)StrideF == this->params.get_kernel_size());
#pragma omp parallel for
                for (unsigned int i = 0; i < this->params.get_kernel_size() * this->params.get_kernel_size(); i++){
                    this->input_array_type.fp[i] = input_array[i] - 128.0f;
                }
            }else if(app_name == "srad_2d"){
                this->input_array_type.fp 
                    = (float*) malloc(this->params.get_kernel_size() * 
                                      this->params.get_kernel_size() * 
                                      sizeof(float));
#pragma omp parallel for
                for (unsigned int i = 0; i < this->params.get_kernel_size() * this->params.get_kernel_size(); i++){
                    this->input_array_type.fp[i] = input_array[i] / 1.;
                }
            }else{
                this->input_array_type.fp  = input_array;
            }
            this->output_array_type.fp = output_array;
        }else{
            std::cout << __func__ << ": app_name: "
                      << app_name << " is not found in any table."
                      << std::endl;
            exit(0);
        }
        timing end = clk::now();
        return get_time_ms(end, start);
    }   

    /* output conversion - search over func_tables to do correct output conversion */
    virtual double output_conversion(){
        timing start = clk::now();
        std::string app_name = this->params.app_name;
        if(if_kernel_in_table(this->func_table_cv, app_name)){
            uint8_t* output_array = 
                reinterpret_cast<uint8_t*>(this->output_array_type.ptr);
            this->output_array_type.mat.convertTo(
                this->output_array_type.mat, 
                CV_8U);
            mat2array(this->output_array_type.mat, output_array);
        }else if(if_kernel_in_table(this->func_table_fp, app_name)){
            if(app_name == "dct8x8_2d"){
                int StrideF = ((int)ceil(this->params.get_kernel_size()/16.0f))*16;
                // ***** float shifting *****
                //AddFloatPlane(128.0f, input, StrideF, ImgSize);
                assert((unsigned)StrideF == this->params.get_kernel_size());
#pragma omp parallel for
                for (unsigned int i = 0; i < this->params.get_kernel_size() * this->params.get_kernel_size(); i++){
                    float tmp = this->output_array_type.fp[i] + 128.0f;
                    this->output_array_type.fp[i] = MIN(MAX(tmp, 0.), 255.);
                }
            }else if(app_name == "srad_2d"){
                //float max_val = FLT_MIN;
//#pragma omp parallel for reduction(max:max_val) 
                //for (unsigned int i = 0; i < this->params.get_kernel_size() * this->params.get_kernel_size(); i++)
                //    max_val = (max_val > this->output_array_type.fp[i])? max_val : this->output_array_type.fp[i];
                //float scale = 255./max_val;
#pragma omp parallel for
                for (unsigned int i = 0; i < this->params.get_kernel_size() * this->params.get_kernel_size(); i++){
                    this->output_array_type.fp[i] = log(this->output_array_type.fp[i]) * 255.;
                }
            }else{
            // no need to convert from float* to float*, pass
            }
        }else{
            // app_name not found in any table. 
        }
        timing end = clk::now();
        return get_time_ms(end, start);
    }

    virtual double run_kernel(unsigned int iter){
        std::string app_name = this->params.app_name;
        if(if_kernel_in_table(this->func_table_cv, app_name)){
            return this->run_kernel_opencv(iter);
        }else if(if_kernel_in_table(this->func_table_fp, app_name)){
            return this->run_kernel_float(iter);
        }else{
            // app_name not found in any table. 
            std::cout << __func__ << ": kernel name: " << app_name 
                      << " not found, program exists." << std::endl;
            std::exit(0);
        }
        return 0.0; // kernel execution is skipped.
    }

private:
    /* opencv type of input/output */
    double run_kernel_opencv(unsigned int iter){
        kernel_existence_checking(this->func_table_cv, this->params.app_name);
        timing start = clk::now();
        for(unsigned int i = 0 ; i < iter ; i++){
            this->func_table_cv[this->params.app_name](this->input_array_type.mat, 
                                                       this->output_array_type.mat);
        }
        timing end = clk::now();
        return get_time_ms(end, start);
    }

    /* float type of input/output */
    double run_kernel_float(unsigned int iter){
        kernel_existence_checking(this->func_table_fp, this->params.app_name);
        timing start = clk::now();
        for(unsigned int i = 0 ; i < iter ; i++){
            this->func_table_fp[this->params.app_name](this->params, 
                                                       this->input_array_type.fp, 
                                                       this->output_array_type.fp);
        }
        timing end = clk::now();
        return get_time_ms(end, start);
    }

    // arrays
    struct ArrayType{
        void* ptr = NULL;
        Mat mat;
        float* fp = NULL;
    };
    Params params;
    ArrayType input_array_type, output_array_type;
    
    // function tables
    typedef void (*func_ptr_opencv)(const Mat, Mat&); // const Mat: input, Mat& : input/output
    typedef void (*func_ptr_float)(Params, float*, float*);
    typedef std::unordered_map<std::string, func_ptr_opencv> func_table_opencv;
    typedef std::unordered_map<std::string, func_ptr_float>  func_table_float;
    func_table_opencv func_table_cv = {
        std::make_pair<std::string, func_ptr_opencv> ("minimum_2d", this->minimum_2d),
        std::make_pair<std::string, func_ptr_opencv> ("sobel_2d", this->sobel_2d),
        std::make_pair<std::string, func_ptr_opencv> ("mean_2d", this->mean_2d),
        std::make_pair<std::string, func_ptr_opencv> ("laplacian_2d", this->laplacian_2d),
        std::make_pair<std::string, func_ptr_opencv> ("kmeans_2d", this->kmeans_2d)
    };
    func_table_float func_table_fp = {
        std::make_pair<std::string, func_ptr_float> ("fft_2d", this->fft_2d),
        std::make_pair<std::string, func_ptr_float> ("dct8x8_2d", this->dct8x8_2d),
        std::make_pair<std::string, func_ptr_float> ("blackscholes_2d", this->blackscholes_2d),
        std::make_pair<std::string, func_ptr_float> ("hotspot_2d", this->hotspot_2d),
        std::make_pair<std::string, func_ptr_float> ("srad_2d", this->srad_2d)
    };

    // kernels
    static void minimum_2d(const Mat in_img, Mat& out_img);
    static void sobel_2d(const Mat in_img, Mat& out_img);
    static void mean_2d(const Mat in_img, Mat& out_img);
    static void laplacian_2d(const Mat in_img, Mat& out_img);
    static void kmeans_2d(const Mat in_img, Mat& out_img);
    static void fft_2d(Params params, float* input, float* output);
    static void dct8x8_2d(Params params, float* input, float* output); 
    static void blackscholes_2d(Params params, float* input, float* output); 
    static void hotspot_2d(Params params, float* input, float* output); 
    static void srad_2d(Params params, float* input, float* output); 
};

#endif
