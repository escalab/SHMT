#include <string>
#include <opencv2/opencv.hpp>
#include "arrays.h"
#include "utils.h"
#include "Common.h"
#include "BmpUtil.h"

/*
    Input array allocation and initialization.
    The data type of arrays depends on applications.
    EX:
        uint8_t: sobel_2d, mean_2d, laplacian_2d
        float:   fft_2d, dct8x8_2d, blackscholes
 */

void init_fft(unsigned int input_total_size, void** input_array){
    srand(2010);
    float* tmp = reinterpret_cast<float*>(*input_array);
    for(unsigned int i = 0 ; i < input_total_size ; i++){
        tmp[i] = (float)(rand() % 16);        
    }
}

void init_srad(Params params, int rows, int cols, void** input_array){
    srand(7);
    Mat in_img;
    read_img(params.input_data_path,
                     rows,
                     cols,
                    in_img);
    in_img.convertTo(in_img, CV_32F);
    mat2array(in_img, (float*)*input_array);
    
    for(int i = 0 ; i < params.problem_size ; i++){
        for(int j = 0 ; j < params.problem_size ; j++){
            ((float*)*input_array)[i*params.problem_size+j] /= 255.; //= rand()/(float)RAND_MAX;
        }
    }
}

void init_dct8x8(Params params, int rows, int cols, void** input_array){
    /* Reference: samples/3_Imaging/dct8x8/dct8x8.cu */
    //char SampleImageFname[256];
    //assert(params.input_data_path.length() < 256);
    //strcpy(SampleImageFname, params.input_data_path.c_str());
    char SampleImageFname[] = "../data/barbara.bmp";
    char *pSampleImageFpath = sdkFindFilePath(SampleImageFname, NULL/*argv[0]*/);
    if (pSampleImageFpath == NULL)
    {
        printf("dct8x8 could not locate Sample Image <%s>\nExiting...\n", pSampleImageFpath);
        exit(EXIT_FAILURE);
    }

    //preload image (acquire dimensions)
    int ImgWidth = rows;
    int ImgHeight = cols;
    ROI ImgSize;
    int res = PreLoadBmp(pSampleImageFpath, &ImgWidth, &ImgHeight);
    ImgSize.width = ImgWidth;
    ImgSize.height = ImgHeight;
    if (res)
    {
        printf("\nError: Image file not found or invalid!\n");
        exit(EXIT_FAILURE);
    }

    //check image dimensions are multiples of BLOCK_SIZE
    if (ImgWidth % BLOCK_SIZE != 0 || ImgHeight % BLOCK_SIZE != 0)
    {
        printf("\nError: Input image dimensions must be multiples of 8!\n");
        exit(EXIT_FAILURE);
    }

    //printf("[%d x %d]... ", ImgWidth, ImgHeight);

    //allocate image buffers
    int ImgStride;
    byte *ImgSrc = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
    //load sample image
    //LoadBmpAsGray(pSampleImageFpath, ImgStride, ImgSize, ImgSrc);
    cv::Mat tmp_img(rows, cols, CV_8U);
    tmp_img = imread(pSampleImageFpath, IMREAD_GRAYSCALE);


    /* ImgSrc has to be resized to [rows x cols] */
    // byte = unsigned char
    byte *ImgSrc_resized = MallocPlaneByte(rows, cols, &ImgStride);
    Mat tmp, resized_tmp;
    array2mat(tmp, ImgSrc, rows, cols);
    Size size = Size(rows, cols);
    resize(tmp_img, resized_tmp, size);
    mat2array(resized_tmp, ImgSrc_resized);
    ImgSize.width = rows;
    ImgSize.height = cols;

    /* Reference: samples/3_Imaging/dct8x8/dct8x8.cu: float WrapperCUDA2() function */   
    //allocate host buffers for DCT and other data
    int StrideF;
    float *ImgF1 = MallocPlaneFloat(rows, cols, &StrideF);

    //convert source image to float representation
    CopyByte2Float(ImgSrc_resized, ImgStride, ImgF1, StrideF, ImgSize);
    //AddFloatPlane(-128.0f, ImgF1, StrideF, ImgSize);
    
    *input_array = ImgF1;

    // assert pixel range in float type
    for(int i = 0 ; i < rows * cols ; i++){
        assert(ImgF1[i] >= 0. && ImgF1[i] <= 255.);
    }
}

void read_hotspot_file(float* vect, int grid_rows, int grid_cols, const char* file){

    int i;//, index;
    FILE *fp;
    int STR_SIZE = 256;
    char str[STR_SIZE];
    float val;
     
    fp = fopen (file, "r");
    if (!fp){
        std::cout << __func__ << ": file could not be opened for reading" << std::endl;
        exit(0);
    }
    for (i=0; i < grid_rows * grid_cols; i++) {
        fgets(str, STR_SIZE, fp);
        if (feof(fp)){
            std::cout << __func__ << ": not enough lines in file" << std::endl;
            exit(0);
        }
        if ((sscanf(str, "%f", &val) != 1) ){
            std::cout << __func__ << ": invalid file format" << std::endl;
            exit(0);
        }
        vect[i] = val;
    }
     
    fclose(fp);
}

void init_hotspot(int rows, int cols, void** input_array){
    float* float_ptr = reinterpret_cast<float*>(*input_array);

    std::string tfile = "../data/hotspot/temp_"+std::to_string(rows);
    std::string pfile = "../data/hotspot/power_"+std::to_string(rows);

    read_hotspot_file(float_ptr, rows, cols, tfile.c_str());
    int offset = rows * cols;
    // concate temp and power arrays into input_array
    read_hotspot_file(&float_ptr[offset], rows, cols, pfile.c_str());

    float temp_max = FLT_MIN;
    float temp_min = FLT_MAX;
    float power_max = FLT_MIN;
    float power_min = FLT_MAX;

    float power_sum = 0.0;

    for(int i = 0 ; i < rows * cols ; i++){
        if(float_ptr[i] > temp_max){
            temp_max = float_ptr[i];
        }
        if(float_ptr[i] < temp_min){
            temp_min = float_ptr[i];
        }
        if(float_ptr[i+offset] > power_max){
            power_max = float_ptr[i+offset];
        }
        if(float_ptr[i+offset] < power_min){
            power_min = float_ptr[i+offset];
        }
        power_sum += float_ptr[i+offset];
    }
}

float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

void init_blackscholes(Params params, 
                       int rows, 
                       int cols, 
                       void** input_array, 
                       void** output_array_baseline, 
                       void** output_array_proposed){
    int OPT_N = rows * cols;
    //printf("...generating input data in CPU mem.\n");
    srand(5347);
    float* h_StockPrice = (float*)*input_array;
    float* h_OptionStrike = &((float*)*input_array)[OPT_N];
    float* h_OptionYears = &((float*)*input_array)[2 * OPT_N];

    float* h_CallResult_baseline = (float*)*output_array_baseline;
    float* h_CallResult_proposed = (float*)*output_array_proposed;
    //float* h_PutResult_baseline = &((float*)*output_array_baseline)[OPT_N];
    //float* h_PutResult_proposed = &((float*)*output_array_proposed)[OPT_N];

    for (int i = 0; i < OPT_N; i++)
    {
        h_CallResult_baseline[i] = 0.0f;
        //h_PutResult_baseline[i]  = -1.0f;
        h_CallResult_proposed[i] = 0.0f;
        //h_PutResult_proposed[i]  = -1.0f;
        h_StockPrice[i]    = RandFloat(5.0f, 30.0f);
        h_OptionStrike[i]  = RandFloat(1.0f, 100.0f);
        h_OptionYears[i]   = RandFloat(0.25f, 10.0f);
    }
}

void data_initialization(Params params,
                         void** input_array,
                         void** output_array_baseline,
                         void** output_array_proposed){
    // TODO: choose corresponding initial depending on app_name.
    int rows = params.problem_size;
    int cols = params.problem_size;
    unsigned int input_total_size = rows * cols;
    unsigned int output_total_size = rows * cols;

    // image filter type of kernels
    if( std::find(params.uint8_t_type_app.begin(), 
                  params.uint8_t_type_app.end(), 
                  params.app_name) !=
        params.uint8_t_type_app.end() ){
        *input_array = (uint8_t*) malloc(input_total_size * sizeof(uint8_t));
        if(params.app_name == "histogram_2d"){
            *output_array_baseline = (int*) malloc(256 * sizeof(int));
            *output_array_proposed = (int*) malloc(256 * sizeof(int));        
            for(int i = 0 ; i < input_total_size ; i++){
                ((uint8_t*)*input_array)[i] = (uint8_t)(rand()%256);
            }
        }else{
            *output_array_baseline = (uint8_t*) malloc(output_total_size * sizeof(uint8_t));
            *output_array_proposed = (uint8_t*) malloc(output_total_size * sizeof(uint8_t));        
            Mat in_img;
            read_img(params.input_data_path,
                     rows,
                     cols,
                    in_img);
            mat2array(in_img, (uint8_t*)*input_array);
            imwrite("bird_input.png", in_img);  
        }
    }else{ // others are default as float type
        if(params.app_name == "fft_2d"){
            *input_array = (float*) malloc(input_total_size * sizeof(float));
            *output_array_baseline = (float*) malloc(output_total_size * sizeof(float));
            *output_array_proposed = (float*) malloc(output_total_size * sizeof(float));   
            init_fft(input_total_size, input_array);
        }else if(params.app_name == "dct8x8_2d"){
            *output_array_baseline = (float*) malloc(output_total_size * sizeof(float));
            *output_array_proposed = (float*) malloc(output_total_size * sizeof(float));   
            init_dct8x8(params, rows, cols, input_array);
        }else if(params.app_name == "hotspot_2d"){
            *output_array_baseline = (float*) malloc(output_total_size * sizeof(float));
            *output_array_proposed = (float*) malloc(output_total_size * sizeof(float));   
            *input_array = (float*) malloc(2 * input_total_size * sizeof(float));   
            init_hotspot(rows, cols, input_array);
        }else if(params.app_name == "blackscholes_2d"){
            *input_array = (float*) malloc(3 * input_total_size * sizeof(float));
            *output_array_baseline = (float*) malloc(/*2 **/ output_total_size * sizeof(float));
            *output_array_proposed = (float*) malloc(/*2 **/ output_total_size * sizeof(float));   
            init_blackscholes(params, rows, cols, input_array, output_array_baseline, output_array_proposed);
        }else if(params.app_name == "srad_2d"){
            *input_array = (float*) malloc(input_total_size * sizeof(float));
            *output_array_baseline = (float*) malloc(output_total_size * sizeof(float));
            *output_array_proposed = (float*) malloc(output_total_size * sizeof(float));   
            init_srad(params, rows, cols, input_array);
        }else{
            *input_array = (float*) malloc(input_total_size * sizeof(float));
            *output_array_baseline = (float*) malloc(output_total_size * sizeof(float));
            *output_array_proposed = (float*) malloc(output_total_size * sizeof(float));   
            Mat in_img;
            read_img(params.input_data_path,
                    rows,
                    cols,
                    in_img);
            in_img.convertTo(in_img, CV_32F);
            mat2array(in_img, (float*)*input_array);
        }
    }
}

/*
    partition array into partitions: allocation and initialization(optional).
*/
void array_partition_initialization(Params params,
                                    bool skip_init,
                                    void** input,
                                    std::vector<void*>& input_pars){
    if(params.app_name == "histogram_2d"){
        if(!skip_init){ // input mat: use this flag to check input mat or output mat
            Mat input_mat, tmp(params.block_size, params.block_size, CV_8U);
            array2mat(input_mat, (uint8_t*)*input, params.problem_size, params.problem_size);
            unsigned int block_total_size = params.block_size * params.block_size;

            // vector of partitions allocation
            input_pars.resize(params.get_block_cnt());   
            for(unsigned int i = 0 ; i < params.get_row_cnt() ; i++){
                for(unsigned int j = 0 ; j < params.get_col_cnt() ; j++){ 
                    unsigned int idx = i * params.get_col_cnt() + j;         
        
                    // partition allocation
                    input_pars[idx] = (uint8_t*) calloc(block_total_size, sizeof(uint8_t));

                    // partition initialization
                    int top_left_w = j*params.block_size;
                    int top_left_h = i*params.block_size;
                    Rect roi(top_left_w, top_left_h, params.block_size, params.block_size); 
                    input_mat(roi).copyTo(tmp); 
                    mat2array(tmp, (uint8_t*)((input_pars[idx])));
                }
            }
        }else{ // output mat
            input_pars.resize(params.get_block_cnt());   
            for(unsigned int i = 0 ; i < params.get_row_cnt() ; i++){
                for(unsigned int j = 0 ; j < params.get_col_cnt() ; j++){ 
                    unsigned int idx = i * params.get_col_cnt() + j;         
        
                    // partition allocation
                    input_pars[idx] = (int*) calloc(256, sizeof(int));
                }
            }
        }
    }else if( std::find(params.uint8_t_type_app.begin(), 
                  params.uint8_t_type_app.end(), 
                  params.app_name) !=
        params.uint8_t_type_app.end() ){
        // prepare for utilizing opencv roi() to do partitioning.
        Mat input_mat, tmp(params.block_size, params.block_size, CV_8U);
        if(!skip_init){
            array2mat(input_mat, (uint8_t*)*input, params.problem_size, params.problem_size);
        }
        unsigned int block_total_size = params.block_size * params.block_size;

        // vector of partitions allocation
        input_pars.resize(params.get_block_cnt());   
        for(unsigned int i = 0 ; i < params.get_row_cnt() ; i++){
            for(unsigned int j = 0 ; j < params.get_col_cnt() ; j++){ 
                unsigned int idx = i * params.get_col_cnt() + j;         
        
                // partition allocation
                input_pars[idx] = (uint8_t*) calloc(block_total_size, sizeof(uint8_t));

                // partition initialization
                if(!skip_init){
                    int top_left_w = j*params.block_size;
                    int top_left_h = i*params.block_size;
                    Rect roi(top_left_w, top_left_h, params.block_size, params.block_size); 
                    input_mat(roi).copyTo(tmp); 
                    mat2array(tmp, (uint8_t*)((input_pars[idx])));
                }
            }
        }
    }else if(params.app_name == "hotspot_2d"){
        // need special partition way to deal with 2 input arrays
        Mat input_mat_temp, input_mat_power;
        Mat tmp_temp(params.block_size, params.block_size, CV_32F);
        Mat tmp_power(params.block_size, params.block_size, CV_32F);
        if(!skip_init){
            array2mat(input_mat_temp,
                      (float*)*input,
                      params.problem_size,
                      params.problem_size);
            array2mat(input_mat_power,
                      &(((float*)(*input))[params.problem_size * params.problem_size]),
                      params.problem_size,
                      params.problem_size);
        }
        unsigned int block_total_size = params.block_size * params.block_size;
 
        // vector of partitions allocation
        input_pars.resize(params.get_block_cnt());
        for(unsigned int i = 0 ; i < params.get_row_cnt() ; i++){
            for(unsigned int j = 0 ; j < params.get_col_cnt() ; j++){
                unsigned int idx = i * params.get_col_cnt() + j;
 
                // partition allocation
                input_pars[idx] = (float*) calloc(2 * block_total_size, sizeof(float));
 
                // partition initialization
                if(!skip_init){
                    int top_left_w = j*params.block_size;
                    int top_left_h = i*params.block_size;
                    Rect roi(top_left_w, top_left_h, params.block_size, params.block_size);
                    input_mat_temp(roi).copyTo(tmp_temp);
                    input_mat_power(roi).copyTo(tmp_power);
                    mat2array(tmp_temp, (float*)((input_pars[idx])));
                    mat2array(tmp_power, &(((float*)(input_pars[idx]))[block_total_size]));
                }
            }
        }
    }else if(params.app_name == "blackscholes_2d"){
        // need special partition way to deal with 3 input arrays
        Mat input_mat_StockPrice, input_mat_OptionStrike, input_mat_OptionYears;
        Mat tmp_StockPrice(params.block_size, params.block_size, CV_32F);
        Mat tmp_OptionStrike(params.block_size, params.block_size, CV_32F);
        Mat tmp_OptionYears(params.block_size, params.block_size, CV_32F);
        if(!skip_init){
            array2mat(input_mat_StockPrice,
                      (float*)*input,
                      params.problem_size,
                      params.problem_size);
            array2mat(input_mat_OptionStrike,
                      &(((float*)(*input))[params.problem_size * params.problem_size]),
                      params.problem_size,
                      params.problem_size);
            array2mat(input_mat_OptionYears,
                      &(((float*)(*input))[2 * params.problem_size * params.problem_size]),
                      params.problem_size,
                      params.problem_size);
        }
        unsigned int block_total_size = params.block_size * params.block_size;
 
        // vector of partitions allocation
        input_pars.resize(params.get_block_cnt());
        for(unsigned int i = 0 ; i < params.get_row_cnt() ; i++){
            for(unsigned int j = 0 ; j < params.get_col_cnt() ; j++){
                unsigned int idx = i * params.get_col_cnt() + j;
 
                // partition allocation
                input_pars[idx] = (float*) calloc(3 * block_total_size, sizeof(float));
 
                // partition initialization
                if(!skip_init){
                    int top_left_w = j*params.block_size;
                    int top_left_h = i*params.block_size;
                    Rect roi(top_left_w, top_left_h, params.block_size, params.block_size);
                    input_mat_StockPrice(roi).copyTo(tmp_StockPrice);
                    input_mat_OptionStrike(roi).copyTo(tmp_OptionStrike);
                    input_mat_OptionYears(roi).copyTo(tmp_OptionYears);
                    mat2array(tmp_StockPrice, (float*)((input_pars[idx])));
                    mat2array(tmp_OptionStrike, &(((float*)(input_pars[idx]))[block_total_size]));
                    mat2array(tmp_OptionYears, &(((float*)(input_pars[idx]))[2 * block_total_size]));
                }
            }
        }
    }else{
        // prepare for utilizing opencv roi() to do partitioning.
        Mat input_mat, tmp(params.block_size, params.block_size, CV_32F);
        if(!skip_init){
            array2mat(input_mat, (float*)*input, params.problem_size, params.problem_size);
        }
        unsigned int block_total_size = params.block_size * params.block_size;

        // vector of partitions allocation
        input_pars.resize(params.get_block_cnt());   
        for(unsigned int i = 0 ; i < params.get_row_cnt() ; i++){
            for(unsigned int j = 0 ; j < params.get_col_cnt() ; j++){ 
                unsigned int idx = i * params.get_col_cnt() + j;         
        
                // partition allocation
                input_pars[idx] = (float*) calloc(block_total_size, sizeof(float));
                
                // partition initialization
                if(!skip_init){
                    int top_left_w = j*params.block_size;
                    int top_left_h = i*params.block_size;
                    Rect roi(top_left_w, top_left_h, params.block_size, params.block_size); 
                    input_mat(roi).copyTo(tmp); 
                    mat2array(tmp, (float*)((input_pars[idx])));
                }
            }
        }
    }
}

/*
    Remap output partitions into one single output array.
*/
void output_array_partition_gathering(Params params,
                                      void** output,
                                      std::vector<void*>& output_pars){

    if(params.app_name == "histogram_2d"){
        for(unsigned int i = 0 ; i < params.get_row_cnt() ; i++){
            for(unsigned int j = 0 ; j < params.get_col_cnt() ; j++){
                unsigned int idx = i * params.get_col_cnt() + j;
                ((int*)*output)[idx] = 0;
                for(int id = 0 ; id < 256 ; id++){
                    ((int*)*output)[id] += ((int*)(output_pars[idx]))[id];
                }
            }
        }
    }else if( std::find(params.uint8_t_type_app.begin(), 
                  params.uint8_t_type_app.end(), 
                  params.app_name) !=
        params.uint8_t_type_app.end() ){
        // prepare for utilizing opencv roi() to do gathering.
        Mat output_mat(params.problem_size, params.problem_size, CV_8U), tmp;
    
        for(unsigned int i = 0 ; i < params.get_row_cnt() ; i++){
            for(unsigned int j = 0 ; j < params.get_col_cnt() ; j++){
                unsigned int idx = i * params.get_col_cnt() + j;
                array2mat(tmp, (uint8_t*)((output_pars[idx])), params.block_size, params.block_size);
                int top_left_w = j*params.block_size;
                int top_left_h = i*params.block_size;
                Rect roi(top_left_w, top_left_h, params.block_size, params.block_size); 
                tmp.copyTo(output_mat(roi));
            }
        }
        mat2array(output_mat, (uint8_t*)*output);
    }else if(params.app_name == "blackscholes_2d"){
        Mat output_mat(/*2 **/ params.problem_size, params.problem_size, CV_32F), tmp_call, tmp_put;
    
        int total_block_size = params.block_size * params.block_size;

        for(unsigned int i = 0 ; i < params.get_row_cnt() ; i++){
            for(unsigned int j = 0 ; j < params.get_col_cnt() ; j++){
                unsigned int idx = i * params.get_col_cnt() + j;
                array2mat(tmp_call, 
                          (float*)(output_pars[idx]), 
                          params.block_size, 
                          params.block_size);
                //array2mat(tmp_put,  
                //          &((float*)(output_pars[idx]))[total_block_size], 
                //          params.block_size, 
                //          params.block_size);
                int top_left_w = j*params.block_size;
                int top_left_h = i*params.block_size;
                Rect roi_call(top_left_w, 
                              top_left_h, 
                              params.block_size, 
                              params.block_size); 
                tmp_call.copyTo(output_mat(roi_call));
                //Rect roi_put(top_left_w, 
                //             top_left_h + params.problem_size, 
                //             params.block_size, 
                //             params.block_size); 
                //tmp_put.copyTo(output_mat(roi_put));
            }
        }
        mat2array(output_mat, (float*)*output);
    }else{
        Mat output_mat(params.problem_size, params.problem_size, CV_32F), tmp;
    
        for(unsigned int i = 0 ; i < params.get_row_cnt() ; i++){
            for(unsigned int j = 0 ; j < params.get_col_cnt() ; j++){
                unsigned int idx = i * params.get_col_cnt() + j;
                array2mat(tmp, (float*)((output_pars[idx])), params.block_size, params.block_size);
                int top_left_w = j*params.block_size;
                int top_left_h = i*params.block_size;
                Rect roi(top_left_w, top_left_h, params.block_size, params.block_size); 
                tmp.copyTo(output_mat(roi));
            }
        }
        mat2array(output_mat, (float*)*output);
    }    
}

void sampling_kernel(Params params, Mat& in, Mat& out, int downsample_block_size){
    auto mode = params.get_sampling_mode();
    if(mode == cv_resize){
        cv::resize(in,
                   out, 
                   cv::Size(downsample_block_size, downsample_block_size), 
                   0, 
                   0, 
                   cv::INTER_LINEAR);
    
    }else{ // other cropping types
        int i_start = 0;
        int j_start = 0;
        if(mode == init_crop){
            i_start = j_start = 0;
        }else if(mode == center_crop){ // assume both square sizes
            i_start = j_start = (in.rows- downsample_block_size)/2;
        }else if(mode == random_crop){
            i_start = int(rand() % (in.rows - downsample_block_size));
            j_start = int(rand() % (in.rows - downsample_block_size));
        }
        int top_left_w = j_start;
        int top_left_h = i_start;
        Rect roi(top_left_w, top_left_h, downsample_block_size, downsample_block_size); 
        in(roi).copyTo(out);
    }
}

template <typename T>
void downsampling_wrapper(Params params,
                          bool skip_init,
                          std::vector<void*> input_pars,
                          std::vector<void*>& input_sampling_pars){
    float rate = params.get_downsampling_rate();
    input_sampling_pars.resize(params.get_block_cnt());

    for(unsigned int i = 0 ; i < params.get_row_cnt() ; i++){
        for(unsigned int j = 0 ; j < params.get_col_cnt() ; j++){
            unsigned int idx = i * params.get_col_cnt() + j;

            unsigned int downsample_block_size = params.block_size * rate;
            unsigned int block_total_size = 
                downsample_block_size * downsample_block_size;

            // downsampling partition allocation
            input_sampling_pars[idx] = 
                (T*) calloc(block_total_size, sizeof(T));
                
            if(!skip_init){
                Mat tmp, sampling_tmp;
                array2mat(tmp,
                          (T*)input_pars[idx],
                          params.block_size,
                          params.block_size);
                array2mat(sampling_tmp, 
                          (T*)input_sampling_pars[idx], 
                          downsample_block_size, 
                          downsample_block_size);
        
                // actual downsampling
                sampling_kernel(params, tmp, sampling_tmp, downsample_block_size);
                
                // store back to input_sampling_pars[idx]
                mat2array(sampling_tmp, (T*)input_sampling_pars[idx]);
            }
        }
    }
}

void array_partition_downsampling(Params params,
                                  bool skip_init,
                                  std::vector<void*> input_pars,
                                  std::vector<void*>& input_sampling_pars){
    if( std::find(params.uint8_t_type_app.begin(), 
                  params.uint8_t_type_app.end(), 
                  params.app_name) !=
        params.uint8_t_type_app.end() ){
        downsampling_wrapper<uint8_t>(params,
                                      skip_init,
                                      input_pars,
                                      input_sampling_pars);
    }else{
        downsampling_wrapper<float>(params,
                                    skip_init,
                                    input_pars,
                                    input_sampling_pars);
    }
}

