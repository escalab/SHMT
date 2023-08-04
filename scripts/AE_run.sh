#!/bin/sh

problem_size=8192
proposed_mode="gt_c-ks" # QAWS-KS
baseline_mode="gpu"

run()
{
	for problem_size in $1
	do
        	./gpgtpu "mean_2d" ${problem_size} 512 $2 ${baseline_mode} ${proposed_mode}  
        	./gpgtpu "sobel_2d" ${problem_size} 512 $2 ${baseline_mode} ${proposed_mode}  
        	./gpgtpu "laplacian_2d" ${problem_size} 2048 $2 ${baseline_mode} ${proposed_mode}  
        	./gpgtpu "fft_2d" ${problem_size} 1024 $2 ${baseline_mode} ${proposed_mode}  
        	./gpgtpu "dct8x8_2d" ${problem_size} 1024 $2 ${baseline_mode} ${proposed_mode}  
        	./gpgtpu "srad_2d" ${problem_size} 512 $2 ${baseline_mode} ${proposed_mode}  
        	./gpgtpu "hotspot_2d" ${problem_size} 2048 $2 ${baseline_mode} ${proposed_mode}  
        	./gpgtpu "dwt_2d" ${problem_size} 2048 $2 ${baseline_mode} ${proposed_mode}  
        	./gpgtpu "blackscholes_2d" ${problem_size} 256 $2 ${baseline_mode} ${proposed_mode}  
        	./gpgtpu "histogram_2d" ${problem_size} 2048 $2 ${baseline_mode} ${proposed_mode}  
	done
}

run ${problem_size} "performance"
run ${problem_size} "quality"
