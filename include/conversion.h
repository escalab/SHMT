#ifndef __CONVERSION_H__
#define __CONVERSION_H__
#include "arrays.h"
#include "params.h"

class UnifyType{
public:
    UnifyType(Params params, void* in);
    ~UnifyType(){};
    uint8_t* get_char_array();
    void save_as_img(const std::string file_name, 
                     unsigned int rows, 
                     unsigned int cols, 
                     void* img);
    void save_as_csv(const std::string file_name,
                     unsigned int rows,
                     unsigned int cols,
                     void* img);
    float* float_array;
    int* int_array;
private:
    Params params;
    uint8_t* char_array;
};

#endif

