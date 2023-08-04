#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

#ifdef __CUDACC__
typedef float2 fComplex;
#else
typedef struct
{
    float x;
    float y;
} fComplex;
#endif

inline int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b); 
};
inline int iAlignUp(int a, int b){
    return (a % b != 0) ? (a - a % b + b) : a;
};

int snapTransformSize(int dataSize);

#endif
