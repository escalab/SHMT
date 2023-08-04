#!/bin/sh

GITTOP="$(git rev-parse --show-toplevel 2>&1)"
BUILD_DIR="./nano_host_build"

cd "${GITTOP}/${BUILD_DIR}"
iter=1
baseline_mode="gpu"

for p in 128 #2 4 8 16 32 64 128
do
    for app_name in  "blackscholes_2d"  #"mean_2d" "sobel_2d" "laplacian_2d" "fft_2d" "dct8x8_2d" "srad_2d" "hotspot_2d" "dwt_2d" "blackscholes_2d" "histogram_2d"
    do
        for problem_size in 8192
        do
            for block_size in 512
            do
                for proposed_mode in "gt_c" #"gt_c-ts" #"gt_c-tu" "gt_c-tr" "gt_c-ks" "gt_c-ku" "gt_c-kr" "gt_c-oracle" #"gt_b" "gt_c-ns-sdev" "gt_c-nr-sdev" "gt_c-ns-range" "gt_c-nr-range" "gt_c-homo"

                do
                    for num in 5 #$(seq -w 0001 0900)
                    #for seq in $(seq -w 0001 0800)
                    do
                        sudo ./gpgtpu ${app_name} ${problem_size} ${block_size} ${iter} ${baseline_mode} ${proposed_mode} ${p} ../data/super${num}.png #../data/ILSVRC/Data/DET/train/ILSVRC2014_train_0000/ILSVRC2014_train_0000${seq}.JPEG  #../data/super${num}.png
                    done
                done
            done
        done
    done
done
