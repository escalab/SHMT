#include "srad.h"
#include "kernels_cpu.h"
#include "kernels_gpu.h"
#include "srad_kernel.cu"

void CpuKernel::srad_2d(Params params, float* input, float* output){
    int rows = params.get_kernel_size();
    int cols = params.get_kernel_size();
    int size_I, size_R, niter = 1, iter;
    float *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI, varROI;

    float Jc, G2, L, num, den, qsqr;
    int *iN, *iS, *jE, *jW, k;
    float *dN, *dS, *dW, *dE;
    float cN, cS, cW, cE, D;

    unsigned int r1 = 0, r2 = rows-1, c1 = 0, c2 = cols-1; // need init
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
                J[k] = J[k] + 0.25*lambda*D;
            }
        }
    }
    free(iN); free(iS); free(jW); free(jE);
    free(dN); free(dS); free(dW); free(dE);
    free(c);
}

void GpuKernel::srad_2d(KernelParams& kernel_params, void** input, void** output){
    int rows = kernel_params.params.get_kernel_size();
    int cols = kernel_params.params.get_kernel_size();
    int size_I, size_R, niter = 1, iter;
    float *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI, varROI;

    float *J_cuda;
    float *C_cuda;
    float *E_C, *W_C, *N_C, *S_C;

    unsigned int r1 = 0, r2 = rows-1, c1 = 0, c2 = cols-1; // need init
    float *c;

    size_I = cols * rows;
    size_R = (r1-r1+1)*(c2-c1+1);
 
    I = (float*)*input;
    J = (float*)*output;
    c = (float *)malloc(sizeof(float)* size_I);

    //Allocate device memory
    cudaMalloc((void**)& J_cuda, sizeof(float)* size_I);
    cudaMalloc((void**)& C_cuda, sizeof(float)* size_I);
    cudaMalloc((void**)& E_C, sizeof(float)* size_I);
    cudaMalloc((void**)& W_C, sizeof(float)* size_I);
    cudaMalloc((void**)& S_C, sizeof(float)* size_I);
    cudaMalloc((void**)& N_C, sizeof(float)* size_I);

    for (int k = 0;  k < size_I; k++ ) {
        J[k] = (float)exp(I[k]) ;
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

        //Currently the input size must be divided by 16 - the block size
        int block_x = cols/BLOCK_SIZE ;
        int block_y = rows/BLOCK_SIZE ;
 
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(block_x , block_y);
 
        //Copy data from main memory to device memory
        cudaMemcpy(J_cuda, J, sizeof(float) * size_I, cudaMemcpyHostToDevice);
 
        //Run kernels
        srad_cuda_1<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda,     cols, rows, q0sqr);
        srad_cuda_2<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda,     cols, rows, lambda, q0sqr);
 
        //Copy data from device memory to main memory
        cudaMemcpy(J, J_cuda, sizeof(float) * size_I, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    //cudaThreadSynchronize();

    cudaFree(C_cuda);
    cudaFree(J_cuda);
    cudaFree(E_C);
    cudaFree(W_C);
    cudaFree(N_C);
    cudaFree(S_C);
    free(c);
}
