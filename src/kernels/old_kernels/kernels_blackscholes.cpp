#include <math.h>
#include <string>
#include <stdio.h>
#include "kernels_cpu.h"
#include "kernels_gpu.h"

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
void CpuKernel::blackscholes_2d(Params params, float* input, float* output){
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

/*
    GPU blackscholes
    Reference: samples/
*/
void GpuKernel::blackscholes_2d(KernelParams& kernel_params, void** in_img, void** out_img){
}


