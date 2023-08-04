#include "utils.h"
#include "quality.h"
#include "math.h"
#include <map>
#include <vector>
#include <float.h>
#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/saliency/saliencySpecializedClasses.hpp>

Quality::Quality(){}

Quality::Quality(std::string app_name,
                 int m, 
                 int n, 
                 int ldn, 
                 int row_blk, 
                 int col_blk,
                 float* input_mat,
                 float* x,
                 float* y,
                 //std::vector<bool> criticality,
                 std::vector<int> proposed_device_type){
    this->app_name     = app_name;
    bool is_hist       = (app_name == "histogram_2d")?true:false; 

    this->in_row = m;
    this->in_col = n;
    this->in_row_blk = row_blk;
    this->in_col_blk = col_blk;
    assert(this->in_row % this->in_row_blk == 0);
    assert(this->in_col % this->in_col_blk == 0);
    this->in_row_cnt = this->in_row / this->in_row_blk;
    this->in_col_cnt = this->in_col / this->in_col_blk;
    this->row          = (is_hist)?1:m;
	this->col          = (is_hist)?256:n;
	this->ldn          = (is_hist)?1:m;
    this->row_blk      = (is_hist)?1:row_blk;
    this->col_blk      = (is_hist)?256:col_blk;
    assert(this->row % this->row_blk == 0);
    assert(this->col % this->col_blk == 0);
    this->row_cnt = this->row / this->row_blk;
    this->col_cnt = this->col / this->col_blk;
    assert(this->row_cnt >= 1);
    assert(this->col_cnt >= 1);
    this->input_mat = input_mat;
    this->target_mat   = x;
	this->baseline_mat = y;
    //this->criticality = criticality;
    this->proposed_device_type = proposed_device_type;
    
    this->result_pars.resize(this->row_cnt * this->col_cnt);
    this->result_critical_pars.resize(this->row_cnt * this->col_cnt);
    if(app_name == "laplacian_2d"){
        histogram_matching(y,
                           x,
                           m,
                           n,
                           row_blk,
                           col_blk,
                           proposed_device_type);
    }

    this->common_kernel(this->result, this->result_critical, 0, 0, this->row, this->col);
/*
    this->common_stats_kernel(this->result.input_dist_stats, 
                              this->input_mat, 
                              0, 
                              0, 
                              this->row, 
                              this->col);
*/
    bool is_tiling = (this->row > this->row_blk)?true:false;

// optional for detailed inspection only    
    if(0 && is_tiling){
        // tiling quality
        for(int i = 0 ; i < this->row_cnt ; i++){
            for(int j = 0 ; j < this->col_cnt ; j++){
                int idx = i * this->col_cnt + j;
                std::cout << "tiling quality(" << i << ", " << j << ")..." << std::endl;
                this->common_kernel(this->result_pars[idx], 
                                    this->result_critical_pars[idx],
                                    i*this->row_blk,
                                    j*this->col_blk,
                                    this->row_blk,
                                    this->col_blk);
              this->common_stats_kernel(this->result_pars[idx].input_dist_stats, 
                                        this->input_mat, 
                                        i*this->row_blk, 
                                        j*this->col_blk, 
                                        this->row_blk, 
                                        this->col_blk);
            }
        }    
    }
}

//static
void Quality::calc_saliency_accuracy(float* in_mat,
                                            int row,
                                            int col,
                                            int row_blk,
                                            int col_blk,
                                            int row_cnt,
                                            int col_cnt,
                                            std::vector<int>proposed_device_type,
                                            float& saliency_ratio, 
                                            float& protected_saliency_ratio,
                                            float& precision/*  TP/(TP+FP)  */){
    cv::Mat mat;
    cv::Mat saliency_map, binary_map;
    std::cout << __func__ << ": start this fun..." << std::endl;
    auto saliency = cv::saliency::StaticSaliencySpectralResidual();
    std::cout << __func__ << ": row: " << row << ",col: " << col << std::endl; 
    array2mat(mat, (float*)in_mat, row, col);

/*
    for(int i = 0 ; i < 10 ; i ++){
        for(int j = 0 ; j < 10 ; j++){
            std::cout << in_mat[i*8192+j] << " ";
        }
        std::cout << std::endl;
    }
*/
    std::cout << __func__ << ": row: " << row << std::endl;
    std::cout << __func__ << ": col: " << col << std::endl;
    std::cout << __func__ << ": row_blk: " << row_blk << std::endl;
    std::cout << __func__ << ": col_blk: " << col_blk << std::endl;
    std::cout << __func__ << ": row_cnt: " << row_cnt << std::endl;
    std::cout << __func__ << ": col_cnt: " << col_cnt << std::endl;
    
    for(auto p: proposed_device_type){
        std::cout << "dev type: " << p << " ";
    }
    std::cout << std::endl;
    
    std::cout << __func__ << ": getting binary map..." << std::endl;
    assert(saliency.computeSaliency(mat, saliency_map));
    assert(saliency.computeBinaryMap(saliency_map, binary_map));
//    imwrite("binary_map.png", binary_map);
    std::cout << __func__ << ": binary_map type: " << binary_map.type() << std::endl;

    unsigned long long int saliency_cnt = 0;
    unsigned long long int saliency_protected_cnt = 0;
    unsigned long long int protected_cnt = 0;

    int GPU_cnt = 0;
    int C_GPU_cnt = 0;

    for(unsigned int i_idx = 0 ; i_idx < row_cnt ; i_idx++){
        for(unsigned int j_idx = 0 ; j_idx < col_cnt ; j_idx++){
            unsigned int i_start = i_idx * row_blk;
            unsigned int j_start = j_idx * col_blk;
            int idx = i_idx*col_cnt+j_idx;

            GPU_cnt += (proposed_device_type[idx] == 2)?1:0;
            
            bool is_salience_block = false;

            for(unsigned int i = i_start ; i < i_start+row_blk ; i++){
                for(unsigned int j = j_start ; j < j_start+col_blk ; j++){
                    bool is_saliency = ((uint8_t)binary_map.at<uint8_t>(i, j))?true:false;
                    if(is_saliency){
                        is_salience_block = true;
                    }
                    saliency_cnt += (is_saliency)?1:0;
                    saliency_protected_cnt += (is_saliency && proposed_device_type[idx] == 2/*gpu*/)?1:0;
                    protected_cnt += (proposed_device_type[idx] == 2)?1:0;
                    
                }
            }
            C_GPU_cnt += (is_salience_block && proposed_device_type[idx] == 2)?1:0;
        }
    }
    saliency_ratio = (float)saliency_cnt / (row * col);
    std::cout << __func__ << ": saliency ratio: " << saliency_cnt << "/" << (row * col) << " (" << saliency_ratio << ")" << std::endl;
    protected_saliency_ratio = (float)saliency_protected_cnt / saliency_cnt;

    std::cout << __func__ << ": protected_saliency_ratio(recall): " << protected_saliency_ratio << std::endl;
    
    std::cout << __func__ << ": precision (TP/(TP+FP)): " << (float)C_GPU_cnt / GPU_cnt 
              << ", (" << C_GPU_cnt << "/" << GPU_cnt << ")" << std::endl;
    precision = (float)C_GPU_cnt / GPU_cnt;
}

void Quality::calc_saliency_accuracy(float& saliency_ratio, float& protected_saliency_ratio){
    cv::Mat mat;
    cv::Mat saliency_map, binary_map;
    auto saliency = cv::saliency::StaticSaliencySpectralResidual();
 
    array2mat(mat, (float*)this->target_mat, this->row, this->col);
    
    std::cout << __func__ << ": getting binary map..." << std::endl;
    assert(saliency.computeSaliency(mat, saliency_map));
    assert(saliency.computeBinaryMap(saliency_map, binary_map));

    unsigned long long int saliency_cnt = 0;
    unsigned long long int saliency_protected_cnt = 0;

    for(unsigned int i_idx = 0 ; i_idx < this->row_cnt ; i_idx++){
        for(unsigned int j_idx = 0 ; j_idx < this->col_cnt ; j_idx++){
            unsigned int i_start = i_idx * this->row_blk;
            unsigned int j_start = j_idx * this->col_blk;
            int idx = i_idx*this->col_cnt+j_idx;
            for(unsigned int i = i_start ; i < i_start+this->row_blk ; i++){
                for(unsigned int j = j_start ; j < j_start+this->col_blk ; j++){
                    bool is_saliency = ((uint8_t)binary_map.at<uint8_t>(i, j))?true:false;
                    saliency_cnt += (is_saliency)?1:0;
                    saliency_protected_cnt += (is_saliency && this->proposed_device_type[idx] == 2/*gpu*/)?1:0;
                }
            }
        }
    }
    saliency_ratio = (float)saliency_cnt / (this->row * this->col);
    std::cout << __func__ << ": saliency ratio: " << saliency_cnt << "/" << (this->row * this->col) << " (" << saliency_ratio << ")" << std::endl;
    protected_saliency_ratio = (float)saliency_protected_cnt / saliency_cnt;
}

void Quality::common_stats_kernel(DistStats& stats, float* x, int i_start, int j_start, int row_size, int col_size){
    float max = FLT_MIN;
    float min = FLT_MAX;
    double sum = 0.0;
	double square_sum = 0.0;
    float entropy = 0.0;
    std::map<float, long int>counts;
    std::map<float, long int>::iterator it;
    it = counts.begin();
    int elements = row_size * col_size;
   
    // max, min, average, entropy(1)
//#pragma omp parallel for reduction(+:sum) reduction(max:max) reduction(min:min)
    for(int i = i_start ; i < i_start+row_size ; i++){
        for(int j = j_start ; j < j_start+col_size ; j++){
            int idx = i*this->ldn+j;
            float tmp = x[idx];
            sum += tmp;
            max = (tmp > max)?tmp:max;
            min = (tmp < min)?tmp:min;
            counts[tmp]++;
        }
    }
    stats.max = max;
    stats.min = min;
	stats.mean = (float)(sum / (double)(elements));
    
    // sdev
#pragma omp parallel for reduction(+:square_sum)
    for(int i = i_start ; i < i_start+row_size ; i++){
        for(int j = j_start ; j < j_start+col_size ; j++){
			square_sum += pow(x[i*this->ldn+j] - stats.mean, 2);
        }
    }
    stats.sdev = pow((float)(square_sum / (double)(elements)), 0.5);

    // entropy(2)
    while(it != counts.end()){
        float p_x = (float)it->second/elements;
        if(p_x > 0){
            entropy-= p_x*log(p_x)/log(2);
        }
        it++;
    }
    stats.entropy = entropy;
}

void Quality::common_kernel(Unit& result, Unit& result_critical, int i_start, int j_start, int row_size, int col_size){
    
    double mse = 0, mse_sum = 0;
	double mean; 
    float baseline_max = FLT_MIN, baseline_min = FLT_MAX;
    float target_max = FLT_MIN, target_min = FLT_MAX;
    double rate = 0, rate_sum = 0;
	int cnt = 0;
	int error_percentage_cnt = 0;
    double baseline_sum = 0.0;

#pragma omp parallel for reduction(+:cnt) reduction(+:error_percentage_cnt) reduction(+:baseline_sum) reduction(max:baseline_max) reduction(max:target_max) reduction(min:baseline_min) reduction(min:target_min) reduction(+:mse_sum) reduction(+:rate_sum)
    for(int i = i_start ; i < i_start+row_size ; i++){
		for(int j = j_start ; j < j_start+col_size ; j++){
			int idx = i*this->ldn+j;
			
            baseline_sum += this->baseline_mat[idx];
            
            baseline_max = 
                (this->baseline_mat[idx] > baseline_max)?
                this->baseline_mat[idx]:
                baseline_max;
            baseline_min = 
                (this->baseline_mat[idx] < baseline_min)?
                this->baseline_mat[idx]:
                baseline_min;
            target_max = 
                (this->target_mat[idx] > target_max)?
                this->target_mat[idx]:
                target_max;
            target_min = 
                (this->target_mat[idx] < target_min)?
                this->target_mat[idx]:
                target_min;
			
            mse_sum += pow(this->target_mat[idx] - this->baseline_mat[idx], 2);
            
            rate_sum += fabs(this->target_mat[idx] - this->baseline_mat[idx]);
            
            if(fabs(this->target_mat[idx] - this->baseline_mat[idx]) > 1e-8){
				error_percentage_cnt++;
			}
			cnt++;
		}
	}

    mse = mse_sum / (cnt);
    rate = rate_sum / (cnt);

    mean = (float)(baseline_sum / (double)(row_size*col_size));

    assert(error_percentage_cnt <= (row_size * col_size));
	
    // SSIM default parameters
	int    L = 255.0; // 2^(# of bits) - 1
	float k1 = 0.01;
	float k2 = 0.03;
	float c1 = 6.5025;  // (k1*L)^2
	float c2 = 58.5225; // (k2*L)^2
/*
    if(target_max > 255.){
        std::cout << __func__ 
                  << ": [WARN] should ignore ssim since array.max = " 
                  << target_max << std::endl;
    }
*/
	// update dynamic range
	L = fabs(target_max - target_min); 
	c1 = (k1*L)*(k1*L);
	c2 = (k2*L)*(k2*L);

	// main calculation
	float ssim = 0.0;
	
	float ux = this->average(this->target_mat, i_start, j_start, row_size, col_size);
	float uy = this->average(this->baseline_mat, i_start, j_start, row_size, col_size);
	float vx = this->sdev(this->target_mat, i_start, j_start, row_size, col_size);
	float vy = this->sdev(this->baseline_mat, i_start, j_start, row_size, col_size);
	float cov = this->covariance(this->target_mat, this->baseline_mat, i_start, j_start, row_size, col_size);

	ssim = ((2*ux*uy+c1) * (2*cov+c2)) / ((pow(ux, 2) + pow(uy, 2) + c1) * (pow(vx, 2) + pow(vy, 2) + c2));
    //assert(ssim >= 0.0 && ssim <= 1.);
    if(ssim < 0.0 || ssim > 1.){
        std::cout << __func__ << " [WARN] ssim is out of bound = "
                  << ssim << ". It may due to wrong value or"
                  << " the result of this benchmark isn't suitable for doing ssim (float type)"
                  << std::endl;
    }
	
    // assign results
    result.rmse = sqrt(mse);
	result.rmse_percentage = (result.rmse/mean) * 100.0;
    result.rate = rate;
    result.error_rate = (rate / mean) * 100.0;
    result.error_percentage = ((float)error_percentage_cnt / (float)(row_size*col_size)) * 100.0;
    result.ssim = ssim;
	result.pnsr = 20*log10(baseline_max) - 10*log10(mse/mean);
}

void Quality::get_minmax(int i_start, 
                         int j_start,
                         int row_size,
                         int col_size, 
                         float* x,
                         float& max, 
                         float& min){
	float curr_max = FLT_MIN;
	float curr_min = FLT_MAX;
	for(int i = i_start ; i < i_start+row_size ; i++){
		for(int j = j_start ; j < j_start+col_size ; j++){
			if(x[i*this->ldn+j] > curr_max){
				curr_max = x[i*this->ldn+j];
			}
			if(x[i*this->ldn+j] < curr_min){
				curr_min = x[i*this->ldn+j];
			}
		}
	}
	max = curr_max;
	min = curr_min;
}

float Quality::max(float* x,
                   int i_start,
                   int j_start,
                   int row_size,
                   int col_size){
    float ret = FLT_MIN;
    for(int i = i_start ; i < i_start+row_size ; i++){
        for(int j = j_start ; j < j_start+col_size ; j++){
            if(x[i*this->ldn+j] > ret){
                ret = x[i*this->ldn+j];
            }
        }
    }
    return ret;
}

float Quality::min(float* x,
                   int i_start,
                   int j_start,
                   int row_size,
                   int col_size){
    float ret = FLT_MAX;
    for(int i = i_start ; i < i_start+row_size ; i++){
        for(int j = j_start ; j < j_start+col_size ; j++){
            if(x[i*this->ldn+j] < ret){
                ret = x[i*this->ldn+j];
            }
        }
    }
    return ret;
}

float Quality::average(float* x, 
                       int i_start,
                       int j_start, 
                       int row_size,
                       int col_size){
	double sum = 0.0;
	for(int i = i_start ; i < i_start+row_size ; i++){
		for(int j = j_start ; j < j_start+col_size ; j++){
			sum += x[i*this->ldn+j];
		}
	}
	return (float)(sum / (double)(row_size*col_size));
}

float Quality::sdev(float* x, int i_start, int j_start, int row_size, int col_size){
	double sum = 0;
	float ux = this->average(x, i_start, j_start, row_size, col_size);
	for(int i = i_start ; i < i_start+row_size ; i++){
		for(int j = j_start ; j < j_start+col_size ; j++){
			sum += pow(x[i*this->ldn+j] - ux, 2);
		}
	}
	return pow((float)(sum / (double)(row_size*col_size)), 0.5);
}

float Quality::entropy(float* x, int i_start, int j_start, int row_size, int col_size){
    float ret = 0.0;
    std::map<float, long int>counts;
    std::map<float, long int>::iterator it;
    for(int i = i_start ; i < i_start+row_size ; i++){
        for(int j = j_start;  j < j_start+col_size ; j++){
            counts[x[i*this->ldn+j]]++;
        }    
    }
    it = counts.begin();
    int elements = row_size * col_size;
    while(it != counts.end()){
        float p_x = (float)it->second/elements;
        if(p_x > 0){
            ret-= p_x*log(p_x)/log(2);
        }
        it++;
    }
    return ret;
}

float Quality::covariance(float* x, float* y, int i_start, int j_start, int row_size, int col_size){
	double sum = 0;
	float ux = this->average(this->target_mat, i_start, j_start, row_size, col_size);
	float uy = this->average(this->baseline_mat, i_start, j_start, row_size, col_size);
	for(int i = i_start ; i < i_start+row_size ; i++){
		for(int j = j_start ; j < j_start+col_size ; j++){
			sum += (x[i*this->ldn+j] - ux) * (y[i*this->ldn+j] - uy);
		}
	}
	return (float)(sum / (double)(row_size*col_size));
}

float Quality::rmse(){
    return this->result.rmse;
}

float Quality::rmse(int i, int j){
    return this->result_pars[i * this->col_cnt + j].rmse;
} 

float Quality::rmse_percentage(){
    return this->result.rmse_percentage;
}

float Quality::rmse_percentage(int i, int j){
    return this->result_pars[i * this->col_cnt + j].rmse_percentage;
} 

float Quality::rate(){
    return this->result.rate;
}

float Quality::error_rate(){
    return this->result.error_rate;
}

float Quality::error_rate(int i, int j){
    return this->result_pars[i * this->col_cnt + j].error_rate;
}

float Quality::error_percentage(){
    return this->result.error_percentage;
}

float Quality::error_percentage(int i, int j){
    return this->result_pars[i*this->col_cnt + j].error_percentage;
}

float Quality::ssim(){
    return this->result.ssim;
}

float Quality::ssim(int i, int j){
    return this->result_pars[i*this->col_cnt + j].ssim;
}

float Quality::pnsr(){
    return this->result.pnsr;
}

float Quality::pnsr(int i, int j){
    return this->result_pars[i*this->col_cnt + j].pnsr;
}

void Quality::print_quality(Unit quality){
    std::cout << "(" << quality.input_dist_stats.max << ", ";
    std::cout << quality.input_dist_stats.min << ", ";
    std::cout << quality.input_dist_stats.mean << ", ";
    std::cout << quality.input_dist_stats.sdev << ", ";
    std::cout << quality.input_dist_stats.entropy << ") | ";
    std::cout << quality.rmse << "\t, ";
    std::cout << quality.rmse_percentage << "\t, ";
    std::cout << quality.error_rate << "\t, ";
    std::cout << quality.error_percentage << "\t, ";
    std::cout << quality.ssim << "\t, ";
    std::cout << quality.pnsr << std::endl;

//    std::fstream myfile;
//    std::string file_path = "./quality.csv";
//    myfile.open(file_path.c_str(), std::ios_base::app);
//    assert(myfile.is_open());
//    myfile << ",," << quality.input_dist_stats.max << ", "
//           << quality.input_dist_stats.min << ", "
//           << quality.input_dist_stats.mean << ", "
//           << quality.input_dist_stats.sdev << ", "
//           << quality.input_dist_stats.entropy << ",,"
//           << quality.rmse << "\t, "
//           << quality.rmse_percentage << "\t, "
//           << quality.error_rate << "\t, "
//           << quality.error_percentage << "\t, "
//           << quality.ssim << "\t, "
//           << quality.pnsr << std::endl;

}

/* print quantized histrogram of mats in integar. */
void Quality::print_histogram(float* input){
    cv::Mat mat, b_hist;
    array2mat(mat, input, this->row, this->col);
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange[] = {range};
    bool uniform = true, accumulate=false;
    calcHist( &mat, 1, 0, cv::Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate );
    std::cout << __func__ << ": hist of mat: " << std::endl;
    for(int i = 0 ; i < histSize ; i++){
        std::cout << b_hist.at<float>(i) << " ";
    }
    std::cout << std::endl;
}

void Quality::print_results(bool is_tiling, int verbose){
    Unit total_quality = {
        this->rmse(),
        this->rmse_percentage(),
        1, // dummy rate
        this->error_rate(),
        this->error_percentage(),
        this->ssim(),
        this->pnsr(),
        {this->result.input_dist_stats.max,
         this->result.input_dist_stats.min,
         this->result.input_dist_stats.mean,
         this->result.input_dist_stats.sdev,
         this->result.input_dist_stats.entropy}
    };
    std::vector<Unit> tiling_quality;

    if(is_tiling == true){
        for(int i = 0 ; i < this->row_cnt ; i++){
            for(int j = 0 ; j < this->col_cnt ; j++){
                int idx = i * this->col_cnt + j;
                Unit per_quality = {
                    this->rmse(i, j),
                    this->rmse_percentage(i, j),
                    1, // dummy rate
                    this->error_rate(i, j),
                    this->error_percentage(i, j),
                    this->ssim(i, j),
                    this->pnsr(i, j),
                    {this->result_pars[idx].input_dist_stats.max,
                     this->result_pars[idx].input_dist_stats.min,
                     this->result_pars[idx].input_dist_stats.mean,
                     this->result_pars[idx].input_dist_stats.sdev,
                     this->result_pars[idx].input_dist_stats.entropy}
                };
                tiling_quality.push_back(per_quality);
            }
        }
    }

    int size = (this->app_name == "histogram_2d")?256:10;

    float baseline_max = FLT_MIN;
    float baseline_min = FLT_MAX;
    float proposed_max = FLT_MIN;
    float proposed_min = FLT_MAX;

    if(verbose){
        std::cout << "baseline result:" << std::endl;

        int first_upper = (this->app_name == "histogram_2d")?1:this->row;
        int sec_upper   = (this->app_name == "histogram_2d")?256:this->row;
        
        for(int i = 0 ; i < first_upper ; i++){
            for(int j = 0 ; j < sec_upper ; j++){
                if(i < size && j < size)
                    std::cout << baseline_mat[i*this->ldn+j] << " ";
                if(baseline_mat[i*this->ldn+j] > baseline_max)
                    baseline_max = baseline_mat[i*this->ldn+j];
                if(baseline_mat[i*this->ldn+j] < baseline_min)
                    baseline_min = baseline_mat[i*this->ldn+j];
            }
            if(i < size && this->app_name != "histogram_2d")
                std::cout << std::endl;
        }
        std::cout << "\nproposed result:" << std::endl;
        for(int i = 0 ; i < first_upper ; i++){
            for(int j = 0 ; j < sec_upper ; j++){
                if(i < size && j < size)
                    std::cout << target_mat[i*this->ldn+j] << " ";
                if(target_mat[i*this->ldn+j] > proposed_max)
                    proposed_max = target_mat[i*this->ldn+j];
                if(target_mat[i*this->ldn+j] < proposed_min)
                    proposed_min = target_mat[i*this->ldn+j];
            }
            if(i < size && this->app_name != "histogram_2d")
                std::cout << std::endl;
        }
    }

    std::cout << "\nbaseline_mat max: " << baseline_max << ", "
              << "min: " << baseline_min << std::endl;
    std::cout << "proposed_mat max: " << proposed_max << ", "
              << "min: " << proposed_min << std::endl;

    printf("=============================================\n");
    printf("Quality results(is_tiling?%d)\n", is_tiling);
    printf("=============================================\n");
    std::cout << "total quality: " << std::endl;

//    std::fstream myfile;
//    std::string file_path = "./quality.csv";
//    myfile.open(file_path.c_str(), std::ios_base::app);
//    assert(myfile.is_open());
    
    std::cout << "input(max, min, mean, sdev, entropy) | rmse,\trmse_percentage(\%),\terror_rate(\%),\terror_percentage(\%),\tssim,\tpnsr(dB)" << std::endl;
//    myfile << "total quality,,,,,,,,,,,," << std::endl;
//    myfile << ",,max, min, mean, sdev, entropy,,\t rmse(\%),\terror_rate(\%),\terror_percentage(\%),\tssim,\tpnsr(dB)" << std::endl;
    print_quality(total_quality);

    if(is_tiling == true){
        std::cout << "tiling quality: " << std::endl;
        std::cout << "(i, j) input(max, min, mean, sdev, entropy) | rmse,\trmse_percentage(\%),\terror_rate(\%),\terror_percentage(\%),\tssim,\tpnsr(dB)" << std::endl;
//        myfile << "tiling quality" << std::endl;
//        myfile << "(i, j), max, min, mean, sdev, entropy,,\t rmse,\trmse_percentage(\%),\terror_rate(\%),\terror_percentage(\%),\tssim,\tpnsr(dB)" << std::endl;
        for(int i = 0 ; i < this->row_cnt  ; i++){
            for(int j = 0 ; j < this->col_cnt  ; j++){
                std::cout << "(" << i << ", " << j << "): ";
                print_quality(tiling_quality[i*this->col_cnt+j]);
            }
        }
    }

    //std::cout << __func__ << ": baseline hist.:" << std::endl;
    //print_histogram(this->baseline_mat);
    //std::cout << __func__ << ": target hist.:" << std::endl;
    //print_histogram(this->target_mat);
    
}

float Quality::static_sdev(uint8_t* array, int num){
    double sum = 0.;
    double square_sum = 0.;
    double mean;
#pragma omp parallel for reduction(+:sum)
    for(int i = 0 ; i < num ; i++){
        sum += array[i];
    }
    mean = sum / num;
#pragma omp parallel for reduction(+:square_sum)
    for(int i = 0 ; i < num ; i++){
        square_sum += pow(array[i] - mean, 2);
    }
    assert(num> 0);
    return (float)pow((double)(square_sum / (double)(num)), 0.5);
}

float Quality::static_sdev(float* array, int num){
    double sum = 0.;
    double square_sum = 0.;
    double mean;
#pragma omp parallel for reduction(+:sum)
    for(int i = 0 ; i < num ; i++){
        sum += array[i];
    }
    mean = sum / num;
#pragma omp parallel for reduction(+:square_sum)
    for(int i = 0 ; i < num ; i++){
        square_sum += pow(array[i] - mean, 2);
    }
    assert(num > 0);
    return (float)pow((double)(square_sum / (double)(num)), 0.5);
}

float Quality::static_sdev(std::vector<float> array){
    double sum = 0.;
    double square_sum = 0.;
    auto const count = static_cast<float>(array.size());
    double mean;
    for(auto p: array){
        sum += p;
    }
    mean = sum / count;
    for(auto p: array){
        square_sum += pow(p - mean, 2);
    }
    assert(count > 0);
    return (float)pow((double)(square_sum / (double)(count)), 0.5);
}

float Quality::static_mean(uint8_t* array, int num){
    double sum = 0.;
#pragma omp parallel for reduction(+:sum)
    for(int i = 0 ; i < num ; i++){
        sum += array[i];
    }
    assert(num > 0);
    return (float) (sum / num);
}

float Quality::static_mean(float* array, int num){
    double sum = 0.;
#pragma omp parallel for reduction(+:sum)
    for(int i = 0 ; i < num ; i++){
        sum += array[i];
    }
    assert(num > 0);
    return (float) (sum / num);
}

float Quality::static_mean(std::vector<float> array){
    double sum = 0.;
    auto const count = static_cast<float>(array.size());
    for(auto p: array){
        sum += p;
    }
    assert(count > 0);
    return (float) (sum / count);
}
