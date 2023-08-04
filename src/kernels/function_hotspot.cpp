#include <string>
#include <cassert>
#include <iostream>

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

extern "C" void read_data(int rows, int cols, float* temp, float* power){
    assert(rows == cols); // only supports square input now
    int internal_rows = (rows >= 256)?rows:256;
    int internal_cols = (cols >= 256)?cols:256;

    std::string tfile = "/home/data/hotspot/temp_" + std::to_string(internal_rows);
    std::string pfile = "/home/data/hotspot/power_" + std::to_string(internal_rows);

    read_hotspot_file(temp, rows, cols, tfile.c_str());
    read_hotspot_file(power, rows, cols, pfile.c_str());
}

extern "C" void hotspot_2d(int rows, int cols, float* input, float* output){
    int num_iterations = 1;

    /* interface */
    int row = rows;
    int col = cols;
    float* result = output;
    float* temp = input;
//    float* power = (float*) malloc(rows * cols * sizeof(float));
    float* power = &input[rows * cols];
    
//    std::string pfile = "/home/data/hotspot/power_" + std::to_string(rows);
//    read_hotspot_file(power, rows, cols, pfile.c_str());

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
