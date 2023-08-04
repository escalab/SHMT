#include "kernels_cpu.h"
#include "kernels_gpu.h"

#include <cuda_runtime.h>

#ifdef RD_WG_SIZE_0_0                                                            
        #define BLOCK_SIZE RD_WG_SIZE_0_0                                        
#elif defined(RD_WG_SIZE_0)                                                      
        #define BLOCK_SIZE RD_WG_SIZE_0                                          
#elif defined(RD_WG_SIZE)                                                        
        #define BLOCK_SIZE RD_WG_SIZE                                            
#else
        #define BLOCK_SIZE 16                                                            
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
    //int BLOCK_SIZE = 16;
    int BLOCK_SIZE_C = BLOCK_SIZE;
    int BLOCK_SIZE_R = BLOCK_SIZE;
    const float amb_temp = 80.0;

    float delta;
    int r, c;
    int chunk;
    int num_chunk = row*col / (BLOCK_SIZE_R * BLOCK_SIZE_C);
    int chunks_in_row = col/BLOCK_SIZE_C;
    int chunks_in_col = row/BLOCK_SIZE_R;

    for( chunk = 0; chunk < num_chunk; ++chunk ){
        int r_start = BLOCK_SIZE_R*(chunk/chunks_in_col);
        int c_start = BLOCK_SIZE_C*(chunk%chunks_in_row);
        int r_end = r_start + BLOCK_SIZE_R > row ? row : r_start + BLOCK_SIZE_R;
        int c_end = c_start + BLOCK_SIZE_C > col ? col : c_start + BLOCK_SIZE_C;

    
        if ( r_start == 0 || c_start == 0 || r_end == row || c_end == col ){
            for ( r = r_start; r < r_start + BLOCK_SIZE_R; ++r ) {
                for ( c = c_start; c < c_start + BLOCK_SIZE_C; ++c ) {
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

        for ( r = r_start; r < r_start + BLOCK_SIZE_R; ++r ) {
            for ( c = c_start; c < c_start + BLOCK_SIZE_C; ++c ) {
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

void CpuKernel::hotspot_2d(Params params, float* input, float* output){

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

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
//#define MIN(a, b) ((a)<=(b) ? (a) : (b))
 
__global__ void calculate_temp(int iteration,  //number of iteration
                               float *power,   //power input
                               float *temp_src,    //temperature input/output
                               float *temp_dst,    //temperature input/output
                               int grid_cols,  //Col of grid
                               int grid_rows,  //Row of grid
                               int border_cols,  // border offset 
                               int border_rows,  // border offset
                               float Cap,      //Capacitance
                               float Rx,
                               float Ry,
                               float Rz,
                               float step,
                               float time_elapsed){

        __shared__ float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float temp_t[BLOCK_SIZE][BLOCK_SIZE]; // saving temparary temperature result
 
    float amb_temp = 80.0;
        float step_div_Cap;
        float Rx_1,Ry_1,Rz_1;

    int bx = blockIdx.x;
        int by = blockIdx.y;
 
    int tx=threadIdx.x;
    int ty=threadIdx.y;
 
    step_div_Cap=step/Cap;

    Rx_1=1/Rx;
    Ry_1=1/Ry;
    Rz_1=1/Rz;
 
        // each block finally computes result for a small block
        // after N iterations. 
        // it is the non-overlapping small blocks that cover 
        // all the input data
 
        // calculate the small block size
    int small_block_rows = BLOCK_SIZE-iteration*2;//EXPAND_RATE
    int small_block_cols = BLOCK_SIZE-iteration*2;//EXPAND_RATE

        // calculate the boundary for the block according to 
        // the boundary of its small block
        int blkY = small_block_rows*by-border_rows;
        int blkX = small_block_cols*bx-border_cols;
        int blkYmax = blkY+BLOCK_SIZE-1;
        int blkXmax = blkX+BLOCK_SIZE-1;

        // calculate the global thread coordination
    int yidx = blkY+ty;
    int xidx = blkX+tx;
 
        // load data if it is within the valid input range
    int loadYidx=yidx, loadXidx=xidx;
        int index = grid_cols*loadYidx+loadXidx;
 
    if(IN_RANGE(loadYidx, 0, grid_rows-1) && IN_RANGE(loadXidx, 0, grid_cols-1)){
            temp_on_cuda[ty][tx] = temp_src[index];  // Load the temperature data from global memory to shared memory
            power_on_cuda[ty][tx] = power[index];// Load the power data from global memory to shared memory
    }
    __syncthreads();

        // effective range within this block that falls within 
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validYmin = (blkY < 0) ? -blkY : 0;
        int validYmax = (blkYmax > grid_rows-1) ? BLOCK_SIZE-1-(blkYmax-grid_rows+1) : BLOCK_SIZE-1;
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > grid_cols-1) ? BLOCK_SIZE-1-(blkXmax-grid_cols+1) : BLOCK_SIZE-1;
 
        int N = ty-1;
        int S = ty+1;
        int W = tx-1;
        int E = tx+1;
 
        N = (N < validYmin) ? validYmin : N;
        S = (S > validYmax) ? validYmax : S;
        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;
 
        bool computed;
        for (int i=0; i<iteration ; i++){
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(tx, validXmin, validXmax) && \
                  IN_RANGE(ty, validYmin, validYmax) ) {
                  computed = true;
                  temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] +
                     (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0*temp_on_cuda[ty][tx]) * Ry_1 +
                     (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0*temp_on_cuda[ty][tx]) * Rx_1 +
                     (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
 
            }
            __syncthreads();
            if(i==iteration-1)
                break;
            if(computed)     //Assign the computation range
                temp_on_cuda[ty][tx]= temp_t[ty][tx];
            __syncthreads();
          }
 
      // update the global memory
      // after the last iteration, only threads coordinated within the 
      // small block perform the calculation and switch on ``computed''
      if (computed){
          temp_dst[index]= temp_t[ty][tx];
      }
}
/*
   compute N time steps
*/
 
int compute_tran_temp(float *MatrixPower,float *MatrixTemp[2], int col, int row, \
        int total_iterations, int num_iterations, int blockCols, int blockRows, int borderCols, int borderRows)
{
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(blockCols, blockRows);
     
    float grid_height = chip_height / row;
    float grid_width = chip_width / col;
     
    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    float Rz = t_chip / (K_SI * grid_height * grid_width);
     
    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float step = PRECISION / max_slope;
    float t;
        float time_elapsed;
    time_elapsed=0.001;
 
        int src = 1, dst = 0;
 
    for (t = 0; t < total_iterations; t+=num_iterations) {
            int temp = src;
            src = dst;
            dst = temp;
            calculate_temp<<<dimGrid, dimBlock>>>(MIN(num_iterations, total_iterations-t), MatrixPower,MatrixTemp[src],MatrixTemp[dst],\
        col,row,borderCols, borderRows, Cap,Rx,Ry,Rz,step,time_elapsed);
    }
        return dst;
}

/* Reference code: rodinia_3.1/cuda/hotspot/hotspot.cu */
void GpuKernel::hotspot_2d(KernelParams& kernel_params, void** input, void** output){

    int dim = kernel_params.params.get_kernel_size();
    int grid_rows = dim;
    int grid_cols = dim;
    int size = dim * dim;

    /* some constants */
    int total_iterations = 1;
    int pyramid_height = 1;

    /* pyramid parameters */
    # define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline
    int borderCols = (pyramid_height)*EXPAND_RATE/2;
    int borderRows = (pyramid_height)*EXPAND_RATE/2;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int smallBlockRow = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
    int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);

    /* host pointers */
    float* host_float_ptr = reinterpret_cast<float*>(*input);
    float* FilesavingTemp = host_float_ptr;
    float* FilesavingPower = &host_float_ptr[size];
    float* MatrixOut = reinterpret_cast<float*>(*output);

    /* device pointers */
    float *MatrixTemp[2], *MatrixPower;
    cudaMalloc((void**)&MatrixTemp[0], sizeof(float)*size);
    cudaMalloc((void**)&MatrixTemp[1], sizeof(float)*size);
    cudaMemcpy(MatrixTemp[0], FilesavingTemp, sizeof(float)*size, cudaMemcpyHostToDevice);
 
    cudaMalloc((void**)&MatrixPower, sizeof(float)*size);
    cudaMemcpy(MatrixPower, FilesavingPower, sizeof(float)*size, cudaMemcpyHostToDevice);
    printf("Start computing the transient temperature\n");
    int ret = compute_tran_temp(MatrixPower,MatrixTemp,grid_cols,grid_rows, \
     total_iterations,pyramid_height, blockCols, blockRows, borderCols, borderRows);
    printf("Ending simulation\n");
    cudaMemcpy(MatrixOut, MatrixTemp[ret], sizeof(float)*size, cudaMemcpyDeviceToHost);
 
    //writeoutput(MatrixOut,grid_rows, grid_cols, ofile);
 
    cudaFree(MatrixPower);
    cudaFree(MatrixTemp[0]);
    cudaFree(MatrixTemp[1]);
}
