#ifndef __QUALITY_H__
#define __QUALITY_H__
#include <stdint.h>
#include <vector>

class Quality{
    public:
    	// default constructor
        Quality(std::string app_name, 
                int row, 
                int col, 
                int ldn, 
                int row_blk, 
                int col_blk, 
                float* input_mat,
                float* target_mat, 
                float* baseline_mat,
                //std::vector<bool> criticality,
                std::vector<int> proposed_device_type);
        Quality();
        /*
            saliency ratio: 
                percentage of critical area / full image area
            protected slaiency ratio:
                percentage of critical area under GPU execution / total critical area
         */
        void calc_saliency_accuracy(float& saliency_ratio, 
                                    float& protected_saliency_ratio);

        // static
        void calc_saliency_accuracy(float* in_mat,
                                           int row,
                                           int col,
                                           int row_blk,
                                           int col_blk,
                                           int row_cnt,
                                           int col_cnt,
                                           std::vector<int> devS_type,
                                           float& saliency_ratio,
                                           float& protected_saliency_ratio,
                                           float& precision);
        
        /* main APIs
	        Now each API call works on one tiling block only,
            and the block is indicated by i_blk_idx, j_blk_idx.
            block sizes are used by this->row_blk, this->col_blk.
         */
        float rmse(); 
        float rmse_percentage();
        float rate();
        float error_rate();
        float error_percentage();
        float ssim();
        float pnsr();
        
        float rmse(int i_idx, int j_idx);
        float rmse_percentage(int i_idx, int j_idx);
        float error_rate(int i_idx, int j_idx);
        float error_percentage(int i_idx, int j_idx);
        float ssim(int i_idx, int j_idx);
        float pnsr(int i_idx, int j_idx);
        
        //float max(float* mat, int i_start, int j_start, int row_size, int col_size);
        float in_max(){ return this->result.input_dist_stats.max; };
        float in_min(){ return this->result.input_dist_stats.min; };
        float in_mean(){ return this->result.input_dist_stats.mean; };
        float in_sdev(){ return this->result.input_dist_stats.sdev; };
        float in_entropy(){ return this->result.input_dist_stats.entropy; };
        
        float in_max(int i_idx, int j_idx){ 
            return this->result_pars[i_idx*this->col_cnt+j_idx].input_dist_stats.max; 
        };
        float in_min(int i_idx, int j_idx){  
            return this->result_pars[i_idx*this->col_cnt+j_idx].input_dist_stats.min; 
        };
        float in_mean(int i_idx, int j_idx){  
            return this->result_pars[i_idx*this->col_cnt+j_idx].input_dist_stats.mean; 
        };
        float in_sdev(int i_idx, int j_idx){  
            return this->result_pars[i_idx*this->col_cnt+j_idx].input_dist_stats.sdev; 
        };
        float in_entropy(int i_idx, int j_idx){ 
            return this->result_pars[i_idx*this->col_cnt+j_idx].input_dist_stats.entropy; 
        };


        void print_results(bool is_tiling, int verbose);
        void print_histogram(float* input);

        int get_row(){ return this->row; }
        int get_col(){ return this->col; }
        int get_row_blk(){ return this->row_blk; }
        int get_col_blk(){ return this->col_blk; }
        int get_row_cnt(){ return this->row_cnt; }
        int get_col_cnt(){ return this->col_cnt; }

        float static_sdev(uint8_t* array, int num);
        float static_sdev(float* array, int num);
        float static_sdev(std::vector<float> array);
        float static_mean(uint8_t* array, int num);
        float static_mean(float* array, int num);
        float static_mean(std::vector<float> array);

    private:
        std::string app_name;

        std::vector<bool> criticality;
        std::vector<int> proposed_device_type;

        struct DistStats{
            float max;
            float min;
            float mean;
            float sdev;
            float entropy;
        };
        
        struct Unit{
            float rmse;
            float rmse_percentage;
            float rate; //for ordering error_rate
            float error_rate;
            float error_percentage;
            float ssim;
            float pnsr;
            
            // input array's statistic
            DistStats input_dist_stats;
        };

        Unit result, result_critical;
        std::vector<Unit> result_pars, result_critical_pars;

        void print_quality(Unit quality);

        // internal helper functions
        void get_minmax(int i_start, 
                        int j_start,
                        int row_size,
                        int col_size,
                        float* x, 
                        float& max,
                        float& min);

        float max(float* mat, int i_start, int j_start, int row_size, int col_size);
        float min(float* mat, int i_start, int j_start, int row_size, int col_size);
        float average(float* mat, int i_start, int j_start, int row_size, int col_size);
        float sdev(float* mat, int i_start, int j_start, int row_size, int col_size);
        float covariance(float* x, float* y, int i_start, int j_start, int row_size, int col_size);
        float entropy(float* mat, int i_start, int j_start, int row_size, int col_size);
        
        // common quality kernel
        void common_kernel(Unit& result, Unit& result_critical, int i_start, int j_start, int row_size, int col_size);
        void common_stats_kernel(DistStats& stats, float* x, int i_start, int j_start, int row_size, int col_size);
        
        int in_row;
        int in_col;
        int in_row_blk;
        int in_col_blk;
        int in_row_cnt;
        int in_col_cnt;

        int row;
	    int col;
	    int ldn;
        int row_blk;
        int col_blk;
        int row_cnt;
        int col_cnt;
        float* input_mat;
        float* target_mat;
	    float* baseline_mat;
};

#endif

