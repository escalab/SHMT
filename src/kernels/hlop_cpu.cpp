#include <math.h>
#include <string>
#include <stdio.h>
#include <cufft.h>
#include <assert.h>
#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <opencv2/cudaarithm.hpp> // addWeighted()
#include <opencv2/cudafilters.hpp> // create[XXX]Filter()
#include <opencv2/highgui/highgui.hpp>
#include "srad.h"
#include "BmpUtil.h"
#include "hlop_cpu.h"
#include "cuda_utils.h"

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
//#define MIN(a, b) ((a)<=(b) ? (a) : (b))

#ifdef RD_WG_SIZE_0_0                                                            
        #define HOTSPOT_BLOCK_SIZE RD_WG_SIZE_0_0                                        
#elif defined(RD_WG_SIZE_0)                                                      
        #define HOTSPOT_BLOCK_SIZE RD_WG_SIZE_0                                          
#elif defined(RD_WG_SIZE)                                                        
        #define HOTSPOT_BLOCK_SIZE RD_WG_SIZE                                            
#else
        #define HOTSPOT_BLOCK_SIZE 16                                                            
#endif

/* some constants */
#define chip_height 0.016
#define chip_width 0.016
#define t_chip 0.0005
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
#define FACTOR_CHIP 0.5
#define MAX_PD 3.0e6

#define BENCHMARK_SIZE 10
#define DCT_BLOCK_SIZE 8
#define DCT_BLOCK_SIZE2 64
#define DCT_BLOCK_SIZE_LOG2 3

float C_a = 1.387039845322148f; //!< a = (2^0.5) * cos(    pi / 16);  Used in forward and inverse DCT.
float C_b = 1.306562964876377f; //!< b = (2^0.5) * cos(    pi /  8);  Used in forward and inverse DCT.
float C_c = 1.175875602419359f; //!< c = (2^0.5) * cos(3 * pi / 16);  Used in forward and inverse DCT.
float C_d = 0.785694958387102f; //!< d = (2^0.5) * cos(5 * pi / 16);  Used in forward and inverse DCT.
float C_e = 0.541196100146197f; //!< e = (2^0.5) * cos(3 * pi /  8);  Used in forward and inverse DCT.
float C_f = 0.275899379282943f; //!< f = (2^0.5) * cos(7 * pi / 16);  Used in forward and inverse DCT.
float C_norm = 0.3535533905932737f; // 1 / (8^0.5)

/**
*  JPEG quality=0_of_12 quantization matrix
*/
float Q_array[DCT_BLOCK_SIZE2] =
{
    32.f,  33.f,  51.f,  81.f,  66.f,  39.f,  34.f,  17.f,
    33.f,  36.f,  48.f,  47.f,  28.f,  23.f,  12.f,  12.f,
    51.f,  48.f,  47.f,  28.f,  23.f,  12.f,  12.f,  12.f,
    81.f,  47.f,  28.f,  23.f,  12.f,  12.f,  12.f,  12.f,
    66.f,  28.f,  23.f,  12.f,  12.f,  12.f,  12.f,  12.f,
    39.f,  23.f,  12.f,  12.f,  12.f,  12.f,  12.f,  12.f,
    34.f,  12.f,  12.f,  12.f,  12.f,  12.f,  12.f,  12.f,
    17.f,  12.f,  12.f,  12.f,  12.f,  12.f,  12.f,  12.f
};
const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;

static double CND(double d)
{   
    const double       A1 = 0.31938153;
    const double       A2 = -0.356563782;
    const double       A3 = 1.781477937;
    const double       A4 = -1.821255978;
    const double       A5 = 1.330274429;
    const double RSQRT2PI = 0.39894228040143267793994605993438;
    
    double
    K = 1.0 / (1.0 + 0.2316419 * fabs(d));
    
    double
    cnd = RSQRT2PI * exp(- 0.5 * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
    
    if (d > 0)
        cnd = 1.0 - cnd;
    
    return cnd;
}

static void BlackScholesBodyCPU(
    float &callResult,
    float &putResult,
    float Sf, //Stock price
    float Xf, //Option strike
    float Tf, //Option years
    float Rf, //Riskless rate
    float Vf  //Volatility rate
)
{   
    double S = Sf, X = Xf, T = Tf, R = Rf, V = Vf;
    
    double sqrtT = sqrt(T);
    double    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);   
    double    d2 = d1 - V * sqrtT;
    double CNDD1 = CND(d1);
    double CNDD2 = CND(d2);
    
    //Calculate Call and Put simultaneously
    double expRT = exp(- R * T);
    callResult   = (float)(S * CNDD1 - X * expRT * CNDD2);
    putResult    = (float)(X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1));
}

/*
    CPU blackscholes
    Reference: samples/
*/
void HLOPCpu::blackscholes_2d(Params params, float* input, float* output){
    int optN = params.get_kernel_size() * params.get_kernel_size();
    for(int opt = 0 ; opt < optN ; opt++){
        BlackScholesBodyCPU(
            //h_CallResult[opt],
            //h_PutResult[opt],
            output[opt],
            output[opt+optN],
            //h_StockPrice[opt],
            //h_OptionStrike[opt],
            //h_OptionYears[opt],
            input[opt],
            input[opt+optN],
            input[opt+2*optN],
            RISKFREE,
            VOLATILITY
        );
    }
}

void SubroutineDCTvector(float *FirstIn, int StepIn, float *FirstOut, int StepOut)
{
    float X07P = FirstIn[0*StepIn] + FirstIn[7*StepIn];
    float X16P = FirstIn[1*StepIn] + FirstIn[6*StepIn];
    float X25P = FirstIn[2*StepIn] + FirstIn[5*StepIn];
    float X34P = FirstIn[3*StepIn] + FirstIn[4*StepIn];

    float X07M = FirstIn[0*StepIn] - FirstIn[7*StepIn];
    float X61M = FirstIn[6*StepIn] - FirstIn[1*StepIn];
    float X25M = FirstIn[2*StepIn] - FirstIn[5*StepIn];
    float X43M = FirstIn[4*StepIn] - FirstIn[3*StepIn];

    float X07P34PP = X07P + X34P;
    float X07P34PM = X07P - X34P;
    float X16P25PP = X16P + X25P;
    float X16P25PM = X16P - X25P;

    FirstOut[0*StepOut] = C_norm * (X07P34PP + X16P25PP);
    FirstOut[2*StepOut] = C_norm * (C_b * X07P34PM + C_e * X16P25PM);
    FirstOut[4*StepOut] = C_norm * (X07P34PP - X16P25PP);
    FirstOut[6*StepOut] = C_norm * (C_e * X07P34PM - C_b * X16P25PM);

    FirstOut[1*StepOut] = C_norm * (C_a * X07M - C_c * X61M + C_d * X25M - C_f * X43M);
    FirstOut[3*StepOut] = C_norm * (C_c * X07M + C_f * X61M - C_a * X25M + C_d * X43M);
    FirstOut[5*StepOut] = C_norm * (C_d * X07M + C_a * X61M + C_f * X25M - C_c * X43M);
    FirstOut[7*StepOut] = C_norm * (C_f * X07M + C_d * X61M + C_c * X25M + C_a * X43M);
}

void computeDCT8x8Gold2(const float *fSrc, float *fDst, int Stride, int width, int height){
    for (int i = 0; i + DCT_BLOCK_SIZE - 1 < height; i += DCT_BLOCK_SIZE)
    {
        for (int j = 0; j + DCT_BLOCK_SIZE - 1 < width; j += DCT_BLOCK_SIZE)
        {
            //process rows
            for (int k = 0; k < DCT_BLOCK_SIZE; k++)
            {
                SubroutineDCTvector((float *)fSrc + (i+k) * Stride + j, 1, fDst + (i+k) * Stride + j, 1);
            }

            //process columns
            for (int k = 0; k < DCT_BLOCK_SIZE; k++)
            {
                SubroutineDCTvector(fDst + i * Stride + (j+k), Stride, fDst + i * Stride + (j+k), Stride);
            }
        }
    }
}

/*Already implemented in utils/BmpUtil.cpp*/
//float round_f(float num)
//{
//    float NumAbs = fabs(num);
//    int NumAbsI = (int)(NumAbs + 0.5f);
//    float sign = num > 0 ? 1.0f : -1.0f;
//    return sign * NumAbsI;
//}

void quantizeGoldFloat(float *fSrcDst, int Stride, int width, int height)
{

    //perform block wise in-place quantization using Q_array
    //Q_array(A) = round(A ./ Q_array) .* Q_array;
    for (int i=0; i<height; i++)
    {
        for (int j=0; j<width; j++)
        {
            int qx = j % DCT_BLOCK_SIZE;
            int qy = i % DCT_BLOCK_SIZE;
            float quantized = round_f(fSrcDst[i*Stride+j] / Q_array[(qy<<DCT_BLOCK_SIZE_LOG2)+qx]);
            fSrcDst[i*Stride+j] = quantized * Q_array[(qy<<DCT_BLOCK_SIZE_LOG2)+qx];
        }
    }
}

void SubroutineIDCTvector(float *FirstIn, int StepIn, float *FirstOut, int StepOut)
{   
    float Y04P   = FirstIn[0*StepIn] + FirstIn[4*StepIn];
    float Y2b6eP = C_b * FirstIn[2*StepIn] + C_e * FirstIn[6*StepIn];
    
    float Y04P2b6ePP = Y04P + Y2b6eP;
    float Y04P2b6ePM = Y04P - Y2b6eP;
    float Y7f1aP3c5dPP = C_f * FirstIn[7*StepIn] + C_a * FirstIn[1*StepIn] + C_c * FirstIn[3*StepIn] + C_d * FirstIn[5*StepIn];
    float Y7a1fM3d5cMP = C_a * FirstIn[7*StepIn] - C_f * FirstIn[1*StepIn] + C_d * FirstIn[3*StepIn] - C_c * FirstIn[5*StepIn];
    
    float Y04M   = FirstIn[0*StepIn] - FirstIn[4*StepIn];
    float Y2e6bM = C_e * FirstIn[2*StepIn] - C_b * FirstIn[6*StepIn];
    
    float Y04M2e6bMP = Y04M + Y2e6bM;
    float Y04M2e6bMM = Y04M - Y2e6bM;
    float Y1c7dM3f5aPM = C_c * FirstIn[1*StepIn] - C_d * FirstIn[7*StepIn] - C_f * FirstIn[3*StepIn] - C_a * FirstIn[5*StepIn];
    float Y1d7cP3a5fMM = C_d * FirstIn[1*StepIn] + C_c * FirstIn[7*StepIn] - C_a * FirstIn[3*StepIn] + C_f * FirstIn[5*StepIn];
    
    FirstOut[0*StepOut] = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
    FirstOut[7*StepOut] = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
    FirstOut[4*StepOut] = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
    FirstOut[3*StepOut] = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);
    
    FirstOut[1*StepOut] = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
    FirstOut[5*StepOut] = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
    FirstOut[2*StepOut] = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
    FirstOut[6*StepOut] = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);
}

void computeIDCT8x8Gold2(const float *fSrc, float *fDst, int Stride, int width, int height)
{
    for (int i = 0; i + DCT_BLOCK_SIZE - 1 < height; i += DCT_BLOCK_SIZE)
    {
        for (int j = 0; j + DCT_BLOCK_SIZE - 1 < width; j += DCT_BLOCK_SIZE)
        {
            //process rows
            for (int k = 0; k < DCT_BLOCK_SIZE; k++)
            {
                SubroutineIDCTvector((float *)fSrc + (i+k) * Stride + j, 1, fDst + (i+k) * Stride + j, 1);
            }

            //process columns
            for (int k = 0; k < DCT_BLOCK_SIZE; k++)
            {
                SubroutineIDCTvector(fDst + i * Stride + (j+k), Stride, fDst + i * Stride + (j+k), Stride);
            }
        }
    }
}

/*
    CPU dct8x8. This implementation requires both dimensions of input/output 
    arrays to be multiply of 8. Incorrect otherwise.
    Reference: samples/3_Imaging/dct8x8/dct8x8.cu: Gold2
*/
void HLOPCpu::dct8x8_2d(Params params, float* input, float* output){
    int width  = params.get_kernel_size();
    int height = params.get_kernel_size();
    int StrideF = width;
    
    float* ImgF2 = (float*) malloc(width * height * sizeof(float));  
    
    assert(width%8==0 && height%8==0);

    for(int i = 0 ; i < BENCHMARK_SIZE ; i++){
        computeDCT8x8Gold2(input, ImgF2, StrideF, width, height);
    }
    quantizeGoldFloat(ImgF2, StrideF, width, height);
    computeIDCT8x8Gold2(ImgF2, output, StrideF, width, height);

//    ROI Size;
//    Size.width  = params.get_kernel_size();
//    Size.height = params.get_kernel_size();
//    AddFloatPlane(128.0f, output, StrideF, Size);
}

/*
    CPU convolveFFT2D, this kernel used a fixed 7x6 convolving kernel.
    Reference: samples/3_Imaging/convolutionFFT2D/convolutionFFT2D_gold.cpp
*/
void HLOPCpu::fft_2d(Params params, float* input, float* output){
    float fft_2d_kernel_array[7*6] = {
        13, 12, 13,  0,  1,  1,
        0,  7,  8,  2,  8,  0,
        5,  9,  1, 11, 11,  3,
        14, 14,  8, 11,  0,  3,
        6,  8, 14, 13,  0, 10,
        10, 11, 14,  1,  2,  0,
        5, 15,  7,  5,  1,  7
    };
    float *h_Result = output;
    float *h_Data = input;
    float *h_Kernel = fft_2d_kernel_array;
    int dataH = params.get_kernel_size();
    int dataW = params.get_kernel_size();
    int kernelH = 7;
    int kernelW = 6;
    int kernelY = 3;
    int kernelX = 4;

    for (int y = 0; y < dataH; y++)
        for (int x = 0; x < dataW; x++)
        {
            double sum = 0;

            for (int ky = -(kernelH - kernelY - 1); ky <= kernelY; ky++)
                for (int kx = -(kernelW - kernelX - 1); kx <= kernelX; kx++)
                {
                    int dy = y + ky;
                    int dx = x + kx;

                    if (dy < 0) dy = 0;

                    if (dx < 0) dx = 0;

                    if (dy >= dataH) dy = dataH - 1;
 
                    if (dx >= dataW) dx = dataW - 1;
                    
                    sum += h_Data[dy * dataW + dx] * h_Kernel[(kernelY - ky) * kernelW + (kernelX - kx)];
                }

            h_Result[y * dataW + x] = (float)sum;
        }
}

void single_iteration(float* result,
                      float* temp,
                      float* power,
                      int row,
                      int col,
                      float Cap_1,
                      float Rx_1,
                      float Ry_1,
                      float Rz_1,
                      float step){
    /* some constants */
    //int HOTSPOT_BLOCK_SIZE = 16;
    int HOTSPOT_BLOCK_SIZE_C = HOTSPOT_BLOCK_SIZE;
    int HOTSPOT_BLOCK_SIZE_R = HOTSPOT_BLOCK_SIZE;
    const float amb_temp = 80.0;

    float delta;
    int r, c;
    int chunk;
    int num_chunk = row*col / (HOTSPOT_BLOCK_SIZE_R * HOTSPOT_BLOCK_SIZE_C);
    int chunks_in_row = col/HOTSPOT_BLOCK_SIZE_C;
    int chunks_in_col = row/HOTSPOT_BLOCK_SIZE_R;

    for( chunk = 0; chunk < num_chunk; ++chunk ){
        int r_start = HOTSPOT_BLOCK_SIZE_R*(chunk/chunks_in_col);
        int c_start = HOTSPOT_BLOCK_SIZE_C*(chunk%chunks_in_row);
        int r_end = r_start + HOTSPOT_BLOCK_SIZE_R > row ? row : r_start + HOTSPOT_BLOCK_SIZE_R;
        int c_end = c_start + HOTSPOT_BLOCK_SIZE_C > col ? col : c_start + HOTSPOT_BLOCK_SIZE_C;

    
        if ( r_start == 0 || c_start == 0 || r_end == row || c_end == col ){
            for ( r = r_start; r < r_start + HOTSPOT_BLOCK_SIZE_R; ++r ) {
                for ( c = c_start; c < c_start + HOTSPOT_BLOCK_SIZE_C; ++c ) {
                    /* Corner 1 */
                    if ( (r == 0) && (c == 0) ) {
                        delta = (Cap_1) * (power[0] +
                            (temp[1] - temp[0]) * Rx_1 +
                            (temp[col] - temp[0]) * Ry_1 +
                            (amb_temp - temp[0]) * Rz_1);
                    }   /* Corner 2 */
                    else if ((r == 0) && (c == col-1)) {
                        delta = (Cap_1) * (power[c] +
                            (temp[c-1] - temp[c]) * Rx_1 +
                            (temp[c+col] - temp[c]) * Ry_1 +
                        (   amb_temp - temp[c]) * Rz_1);
                    }   /* Corner 3 */
                    else if ((r == row-1) && (c == col-1)) {
                        delta = (Cap_1) * (power[r*col+c] +
                            (temp[r*col+c-1] - temp[r*col+c]) * Rx_1 +
                            (temp[(r-1)*col+c] - temp[r*col+c]) * Ry_1 +
                        (   amb_temp - temp[r*col+c]) * Rz_1);
                    }   /* Corner 4 */
                    else if ((r == row-1) && (c == 0)) {
                        delta = (Cap_1) * (power[r*col] +
                            (temp[r*col+1] - temp[r*col]) * Rx_1 +
                            (temp[(r-1)*col] - temp[r*col]) * Ry_1 +
                            (amb_temp - temp[r*col]) * Rz_1);
                    }   /* Edge 1 */
                    else if (r == 0) {
                        delta = (Cap_1) * (power[c] +
                            (temp[c+1] + temp[c-1] - 2.0*temp[c]) * Rx_1 +
                            (temp[col+c] - temp[c]) * Ry_1 +
                            (amb_temp - temp[c]) * Rz_1);
                    }   /* Edge 2 */
                    else if (c == col-1) {
                        delta = (Cap_1) * (power[r*col+c] +
                            (temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0*temp[r*col+c]) * Ry_1 +
                            (temp[r*col+c-1] - temp[r*col+c]) * Rx_1 +
                            (amb_temp - temp[r*col+c]) * Rz_1);
                    }   /* Edge 3 */
                    else if (r == row-1) {
                        delta = (Cap_1) * (power[r*col+c] +
                            (temp[r*col+c+1] + temp[r*col+c-1] - 2.0*temp[r*col+c]) * Rx_1 +
                            (temp[(r-1)*col+c] - temp[r*col+c]) * Ry_1 +
                            (amb_temp - temp[r*col+c]) * Rz_1);
                    }   /* Edge 4 */
                    else if (c == 0) {
                        delta = (Cap_1) * (power[r*col] +
                            (temp[(r+1)*col] + temp[(r-1)*col] - 2.0*temp[r*col]) * Ry_1 +    
                            (temp[r*col+1] - temp[r*col]) * Rx_1 +
                            (amb_temp - temp[r*col]) * Rz_1);
                    }
                    result[r*col+c] =temp[r*col+c]+ delta;    
                }
            }
            continue;
        }

        for ( r = r_start; r < r_start + HOTSPOT_BLOCK_SIZE_R; ++r ) {
            for ( c = c_start; c < c_start + HOTSPOT_BLOCK_SIZE_C; ++c ) {
            /* Update Temperatures */
                result[r*col+c] =temp[r*col+c]+
                     ( Cap_1 * (power[r*col+c] +
                    (temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.f*temp[r*col+c]) * Ry_1 +
                    (temp[r*col+c+1] + temp[r*col+c-1] - 2.f*temp[r*col+c]) * Rx_1 +
                    (amb_temp - temp[r*col+c]) * Rz_1));
            }
        }
    }
}

void HLOPCpu::hotspot_2d(Params params, float* input, float* output){

    int num_iterations = 1;

    /* interface */
    int row = params.get_kernel_size();
    int col = params.get_kernel_size();
    float* temp = input;
    float* power = &input[row*col]; // second half of concated input array
    float* result = output;

    float grid_height = chip_height / row;
    float grid_width  = chip_width  / col;
    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    float Rz = t_chip / (K_SI * grid_height * grid_width);

    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float step = PRECISION / max_slope / 1000.0;

    float Rx_1=1.f/Rx;
    float Ry_1=1.f/Ry;
    float Rz_1=1.f/Rz;
    float Cap_1 = step/Cap;

    //int array_size = row * col;

    float* r = result;
    float* t = temp;
    for (int i = 0; i < num_iterations ; i++){
        single_iteration(r, t, power, row, col, Cap_1, Rx_1, Ry_1, Rz_1, step);
        float* tmp = t;
        t = r;
        r = tmp;
    }
}

void HLOPCpu::kmeans_2d(const Mat in_img, Mat& out_img){
    int k = 4;
    std::vector<int> labels;
    cv::Mat1f centers;
    unsigned int size = in_img.rows * in_img.cols;
    cv::Mat in_tmp = in_img.reshape(1, size);
    in_tmp.convertTo(in_tmp, CV_32F);
    cv::kmeans(in_tmp, // kmeans only takes CV_32F data type 
               k, 
               labels, 
               cv::TermCriteria(TermCriteria::MAX_ITER/*|TermCriteria::EPS*/, 
                                10, // max iteration 
                                1.0), // epsilon
               3, // attempts
               cv::KMEANS_PP_CENTERS,
               centers);
    for (unsigned int i = 0; i < size; i++) {
        in_tmp.at<float>(i) = centers(labels[i]);
    }
    out_img = in_tmp.reshape(1, in_img.rows);
    out_img.convertTo(out_img, CV_8U);
}

void HLOPCpu::laplacian_2d(const Mat in_img, Mat& out_img){
    int ddepth = CV_32F;
    Laplacian(in_img, out_img, ddepth, 3/*kernel size*/, 1/*scale*/, 0/*delta*/, BORDER_DEFAULT);
    convertScaleAbs(out_img, out_img);
}

void HLOPCpu::mean_2d(const Mat in_img, Mat& out_img){
    blur(in_img, out_img, Size(3, 3), Point(-1, -1), BORDER_DEFAULT);
}

/* A dummy kernel for testing only. */
void HLOPCpu::minimum_2d(const Mat in_img, Mat& out_img){
    out_img = in_img;
}

void HLOPCpu::sobel_2d(const Mat in_img, Mat& out_img){
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    int ddepth = CV_32F; // CV_8U, CV_16S, CV_16U, CV_32F, CV_64F
    Sobel(in_img, grad_x, ddepth, 1, 0, 3/*ksize*/, 1/*scale*/, 0/*delta*/, BORDER_DEFAULT);
    Sobel(in_img, grad_y, ddepth, 0, 1, 3/*ksize*/, 1/*scale*/, 0/*delta*/, BORDER_DEFAULT);

    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, out_img);
}

void HLOPCpu::srad_2d(Params params, float* input, float* output){
    int rows = params.get_kernel_size();
    int cols = params.get_kernel_size();
    int size_I, size_R, niter = 10, iter;
    float *I, *J, lambda=0.5, q0sqr, sum, sum2, tmp, meanROI, varROI;

    float Jc, G2, L, num, den, qsqr;
    int *iN, *iS, *jE, *jW, k;
    float *dN, *dS, *dW, *dE;
    float cN, cS, cW, cE, D;

    unsigned int r1 = 0, r2 = 127/*rows-1*/, c1 = 0, c2 = 127/*cols-1*/; // need init
    float *c;

    size_I = cols * rows;
    size_R = (r1-r1+1)*(c2-c1+1);

    I = input;
    J = output;
    c = (float *)malloc(sizeof(float)* size_I);

    iN = (int *)malloc(sizeof(unsigned int*) * rows) ;
    iS = (int *)malloc(sizeof(unsigned int*) * rows) ;
    jW = (int *)malloc(sizeof(unsigned int*) * cols) ;
    jE = (int *)malloc(sizeof(unsigned int*) * cols) ;
 
    dN = (float *)malloc(sizeof(float)* size_I) ;
    dS = (float *)malloc(sizeof(float)* size_I) ;
    dW = (float *)malloc(sizeof(float)* size_I) ;
    dE = (float *)malloc(sizeof(float)* size_I) ;

    for (int i=0; i< rows; i++) {
        iN[i] = i-1;
        iS[i] = i+1;
    }
    for (int j=0; j< cols; j++) {
        jW[j] = j-1;
        jE[j] = j+1;
    }
    iN[0]    = 0;
    iS[rows-1] = rows-1;
    jW[0]    = 0;
    jE[cols-1] = cols-1;

    for (int k = 0;  k < size_I; k++ ) {
        J[k] = (float)exp(I[k]) ;
        //printf("I[%d]: %f, J[%d]: %f\n", k, I[k], k, J[k]);
        //getchar();
    }

    for(iter=0; iter < niter ; iter++){
        sum=0; sum2=0;
        for (int i=r1; i<=r2; i++) {
            for (int j=c1; j<=c2; j++) {
                tmp   = J[i * cols + j];
                sum  += tmp ;
                sum2 += tmp*tmp;
            }
        }
        meanROI = sum / size_R;
        varROI  = (sum2 / size_R) - meanROI*meanROI;
        q0sqr   = varROI / (meanROI*meanROI);
        
        for (int i = 0 ; i < rows ; i++) {
            for (int j = 0; j < cols; j++) {
                k = i * cols + j;
                Jc = J[k];

                // directional derivates
                dN[k] = J[iN[i] * cols + j] - Jc;
                dS[k] = J[iS[i] * cols + j] - Jc;
                dW[k] = J[i * cols + jW[j]] - Jc;
                dE[k] = J[i * cols + jE[j]] - Jc;
                 
                G2 = (dN[k]*dN[k] + dS[k]*dS[k]
                    + dW[k]*dW[k] + dE[k]*dE[k]) / (Jc*Jc);
                 
                L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;
                 
                num  = (0.5*G2) - ((1.0/16.0)*(L*L)) ;
                den  = 1 + (.25*L);
                qsqr = num/(den*den);

                // diffusion coefficent (equ 33)
                den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
                
                c[k] = 1.0 / (1.0+den) ;
 
                // saturate diffusion coefficent
                if (c[k] < 0) {c[k] = 0;}
                else if (c[k] > 1) {c[k] = 1;}
            }
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
 
                // current index
                k = i * cols + j;

                // diffusion coefficent
                cN = c[k];
                cS = c[iS[i] * cols + j];
                cW = c[k];
                cE = c[i * cols + jE[j]];
 
                // divergence (equ 58)
                D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];

                // image update (equ 61)
                //std::cout << __func__ << ": D: " << D << std::endl;
                J[k] = J[k] + 0.25*lambda*D;
            }
        }
    }
    free(iN); free(iS); free(jW); free(jE);
    free(dN); free(dS); free(dW); free(dE);
    free(c);
}

