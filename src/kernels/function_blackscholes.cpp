#include <math.h>
#include <string>
#include <stdio.h>
#include <assert.h>
#include <iostream>

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
    //float &putResult,
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
    //putResult    = (float)(X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1));
}

/*
    CPU blackscholes
    Reference: samples/
*/
extern "C" void blackscholes_2d(int row, int col, float* input, float* output){
    int optN = row * col;
    for(int opt = 0 ; opt < optN ; opt++){
        BlackScholesBodyCPU(
	    // return call only
            output[opt], // call
//            output[opt+optN], // put
            input[opt],
            input[opt+optN],
            input[opt+2*optN],
            RISKFREE,
            VOLATILITY
        );
    }
}

float RandFloat(float low, float high)
{   
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

extern "C" void read_data(int row, 
		          int col, 
			  float* h_StockPrice, 
			  float* h_OptionStrike, 
			  float* h_OptionYears){
    int OPT_N = row * col;
    srand(5347);

    //input_array  = (float*) malloc(OPT_N * 3 * sizeof(float));
    //output_array = (float*) malloc(OPT_N * 2 * sizeof(float)); 

    //float* h_StockPrice = (float*)input_array;
    //float* h_OptionStrike = &((float*)input_array)[OPT_N];
    //float* h_OptionYears = &((float*)input_array)[2 * OPT_N];

    //float* h_CallResult = (float*)output_array;
    //float* h_PutResult  = &((float*)output_array)[OPT_N];

    for (int i = 0; i < OPT_N; i++)
    {
        //h_CallResult[i] = 0.0f;
        //h_PutResult[i]  = -1.0f;
        h_StockPrice[i]    = RandFloat(5.0f, 30.0f);
        h_OptionStrike[i]  = RandFloat(1.0f, 100.0f);
        h_OptionYears[i]   = RandFloat(0.25f, 10.0f);
    }
}
