#include <fstream>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "conversion.h"

using namespace cv;

UnifyType::UnifyType(Params params, void* in){
    this->params = params;
    if(params.app_name == "histogram_2d"){
        int* int_tmp = reinterpret_cast<int*>(in);
        this->float_array = (float*) malloc(256 * sizeof(float));
        for(int i = 0 ; i < 256; i++){
                this->float_array[i] = int_tmp[i]; // int to float conversion
        }
    }else if( std::find(params.uint8_t_type_app.begin(),
                  params.uint8_t_type_app.end(),
                  params.app_name) !=
        params.uint8_t_type_app.end() ){
        uint8_t* tmp = reinterpret_cast<uint8_t*>(in);
        this->float_array = (float*) malloc(params.problem_size * 
                                            params.problem_size * 
                                            sizeof(float));
#pragma omp parallel for 
        for(int i = 0 ; i < params.problem_size * params.problem_size; i++){
                this->float_array[i] = tmp[i]; // uint8_t to float conversion
        }
        this->char_array = tmp; // record the uint8_t pointer
    }else{ // others are default as float type
        this->float_array = reinterpret_cast<float*>(in);
    }
}

void UnifyType::save_as_img(const std::string file_name, 
                            unsigned int rows, 
                            unsigned int cols, 
                            void* img){
    if( std::find(this->params.uint8_t_type_app.begin(),
                  this->params.uint8_t_type_app.end(),
                  this->params.app_name) !=
        this->params.uint8_t_type_app.end() ){
        Mat mat(rows, cols, CV_8U);
        array2mat(mat, this->char_array, rows, cols);
        assert(!mat.empty());
        imwrite(file_name.c_str(), mat);
    }else if(this->params.app_name == "hotspot_2d"){
        std::cout << __func__ << ": saved image is only for visualization, "
                  << "and it doesn't reflect actual value(s)." << std::endl;
        Mat mat(rows, cols, CV_8U);
        uint8_t* tmp = (uint8_t*) malloc(rows * cols * sizeof(uint8_t));
        float min = 322.;
        float range = 22.;
#pragma omp parallel for
        for(unsigned int i = 0 ; i < rows * cols ; i++){
            tmp[i] = (uint8_t)((this->float_array[i] - min) * (255./range));
        }
        array2mat(mat, tmp, rows, cols);
        assert(!mat.empty());
        imwrite(file_name.c_str(), mat);
    }else if(this->params.app_name == "dct8x8_2d"){
        uint8_t* tmp = (uint8_t*) malloc(rows * cols * sizeof(uint8_t));
#pragma omp parallel for
        for(int i = 0 ; i < rows * cols ; i++){
            tmp[i] = (uint8_t)this->float_array[i];
        }
        Mat mat(rows, cols, CV_8U);
        // TODO: how to show float array?
        array2mat(mat, tmp, rows, cols);
        assert(!mat.empty());
        imwrite(file_name.c_str(), mat);
    }else if(this->params.app_name == "srad_2d"){
        uint8_t* tmp = (uint8_t*) malloc(rows * cols * sizeof(uint8_t));
#pragma omp parallel for
        for(int i = 0 ; i < rows * cols ; i++){
            tmp[i] = (uint8_t)(this->float_array[i] /** 255.*/);
        }
        Mat mat(rows, cols, CV_8U);
        // TODO: how to show float array?
        array2mat(mat, tmp, rows, cols);
        assert(!mat.empty());
        imwrite(file_name.c_str(), mat);
    }else{
        std::cout << __func__ << ": unprocessed float type image." << std::endl;
        Mat mat(rows, cols, CV_32F);
        // TODO: how to show float array?
        array2mat(mat, this->float_array, rows, cols);
        assert(!mat.empty());
        imwrite(file_name.c_str(), mat);
    }
}

void UnifyType::save_as_csv(const std::string file_name,
                            unsigned int rows,
                            unsigned int cols,
                            void* img){
    std::fstream myfile;
    myfile.open(file_name.c_str(), std::ios_base::out | std::ios::binary);
    assert(myfile.is_open());

    if(std::find(this->params.uint8_t_type_app.begin(),
                this->params.uint8_t_type_app.end(),
                this->params.app_name) !=
       this->params.uint8_t_type_app.end() ){
        uint8_t* tmp = reinterpret_cast<uint8_t*>(img);    
        for(unsigned int i = 0 ; i < rows ; i++){
            for(unsigned int j = 0 ; j < cols ; j++){
                myfile << std::hex << std::setfill('0') << std::setw(2) 
                       << (unsigned)tmp[i*cols+j] << ",";        
            }
            myfile << std::endl;
        }    
    }else if(this->params.app_name == "dct8x8_2d"){
        float* tmp = reinterpret_cast<float*>(img);    
        for(unsigned int i = 0 ; i < rows ; i++){
            for(unsigned int j = 0 ; j < cols ; j++){
                myfile << (uint8_t)tmp[i*cols+j] << ",";        
            }
            myfile << std::endl;
        }    
    }else{
        float* tmp = reinterpret_cast<float*>(img);    
        for(unsigned int i = 0 ; i < rows ; i++){
            for(unsigned int j = 0 ; j < cols ; j++){
                myfile << tmp[i*cols+j] << ",";        
            }
            myfile << std::endl;
        }    
    }

}


