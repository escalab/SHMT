#!/bin/sh

g++ -fPIC -shared -o function_hotspot.so ./function_hotspot.cpp 
g++ -fPIC -shared -o function_srad.so ./function_srad.cpp -I./include
g++ -fPIC -shared -o function_blackscholes.so ./function_blackscholes.cpp 
nvcc --compiler-options "-fPIC" -shared -o function_dwt.so -I/usr/local/cuda/include -I./include/ ./function_dwt.cu ./fdwt97.cu ./rdwt97.cu
