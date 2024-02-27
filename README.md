# SHMT
The Simultaneous and Heterogenous Multithreading project (Jetson nano + edgeTPU). \
This project contains two major parts: kernel model training scheme and the actual partition execution scheme. These two schemes require two different corresponding platforms. 

[![DOI](https://zenodo.org/badge/674087894.svg)](https://zenodo.org/badge/latestdoi/674087894)


## Tested platforms
The kernel model training stage: Ubuntu 20.04 x86_64 12th Gen Intel(R) Core(TM) i9-12900KF (Rayquaza in Escal's environment) \
The partition execution stage: Ubuntu 18.04 aarch64 Cortex-A57 (nano-2 in Escal's environment)


## Pre-requisites
1. NVIDIA Drivers (CUDA 10.2)

## Setup
### 1. Install docker
Please refer to: https://www.simplilearn.com/tutorials/docker-tutorial/how-to-install-docker-on-ubuntu \
or the official website: https://docs.docker.com/engine/install/ubuntu/

```
sudo apt-get remove docker docker-engine docker.io
sudo apt-get update
sudo apt install docker.io
sudo snap install docker
```
Check docker version
```
docker --version
```
Run docker hello-world to verify
```
sudo docker run hello-world
```

### 2. Install NVIDIA Container Toolkit 
Please refer to: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

1. install ```curl```
```
sudo apt update && sudo apt upgrade
sudo apt install curl
```
2. install the NVIDIA Container Toolkit
``` 
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
&& curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### (Optional) Install opencv2 (if nvidia-docker is not used)
Reference : https://qengineering.eu/install-opencv-4.5-on-jetson-nano.html

## Build and execution
### 1. The kernel model training mode (on Rayquaza as example)
Please refer to src/Python/generate_kernel_model.py for more details.

### 2. The partition execution mode (on Jetson nano-2 as example)
```
$ git clone https://github.com/escal/SHMT
$ sh scripts/docker_setup_partition.sh
$ sh scripts/docker_launch_partition.sh
(docker)$ mkdir build
(docker)$ cd build
(docker)$ cmake ..
(docker)$ make -j4
(docker)$ cd scripts/
(docker)$ sh AE_run.sh
```

### (Optional) To build the gptpu_utils shared library from source: (is provided under /lib/aarch64)
1. Cross-compile it on a x86 machine
```
(on x86 host at $GITTOP)$ sh scripts/build_gptpu_utils_on_x86.sh
```
2. install the output ```libgptpu_utils.so``` and header ```gptpu_utils.h``` on nano aarch64 machine
```
(on aarch64 host at $GITTOP)$ sh scripts/install_gptpu_utils_on_aarch64.sh
```

## Trouble shooting
### 1. ```docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].``` 
1. Follow this steps to uninstall and install docker for nvidia image: https://github.com/NVIDIA/nvidia-docker/issues/1637#issuecomment-1130151618. 
2. Make sure the following command gives good ```nvidia-smi``` output: \
```sudo docker run --rm --gpus all nvidia/cuda:11.7.0-devel-ubuntu20.04 nvidia-smi``` \
(Replace version numbers accordingly if cuda and Ubuntu versions vary.)

### 2. ```scripts/docker_setup.sh: 7: .: Can't open fatal: unsafe repository ('/home/kuanchiehhsu/GPGTPU' is owned by someone else)```
To add an exception for this directory, call:
```
git config --global --add safe.directory '*'
```
reference: https://stackoverflow.com/questions/71901632/fatal-error-unsafe-repository-home-repon-is-owned-by-someone-else

### 3. ```Unsupported data type in custom op handler``` during run time.
One of many reasons is that too many edgetpu runtime versions are installed as mentioned here: https://github.com/google-coral/tflite/issues/45#issuecomment-815080437 \
Remove all ```dpkg -l | grep edgetpu``` listed ones and make sure that the one compiled from source code within this project ```libedgetpu``` is used.

### 4. ```fatal error: opencv2/cudaarithm.hpp: No such file or directory```
opencv2 is missing (not installed yet), please install first. \
Basic summary (using installing 4.5.5 version as example):
```
# download the latest version
$ cd ~
$ wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.5.zip
$ wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.5.zip
# unpack
$ unzip opencv.zip
$ unzip opencv_contrib.zip
# some administration to make live easier later on
$ mv opencv-4.5.5 opencv
$ mv opencv_contrib-4.5.5 opencv_contrib
# clean up the zip files
$ rm opencv.zip
$ rm opencv_contrib.zip
```
```
$ cd ~/opencv
$ mkdir build
$ cd build
```
(Note: CMAKE_INSTALL_PREFIX typical preference: /usr/local/)
```
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr \
-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
-D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
-D WITH_OPENCL=OFF \
-D WITH_CUDA=ON \
-D CUDA_ARCH_BIN=5.3 \
-D CUDA_ARCH_PTX="" \
-D WITH_CUDNN=ON \
-D WITH_CUBLAS=ON \
-D ENABLE_FAST_MATH=ON \
-D CUDA_FAST_MATH=ON \
-D OPENCV_DNN_CUDA=ON \
-D ENABLE_NEON=ON \
-D WITH_QT=OFF \
-D WITH_OPENMP=ON \
-D BUILD_TIFF=ON \
-D WITH_FFMPEG=ON \
-D WITH_GSTREAMER=ON \
-D WITH_TBB=ON \
-D BUILD_TBB=ON \
-D BUILD_TESTS=OFF \
-D WITH_EIGEN=ON \
-D WITH_V4L=ON \
-D WITH_LIBV4L=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D BUILD_EXAMPLES=OFF ..
```
(Note: the make step might take 1.5 hours on Jetson nano)
```
$ make -j4
```

### 5. ```failed to create shim: OCI runtime create failed: failed to create NVIDIA Container Runtime: failed to construct OCI spec modifier: failed to construct discoverer: failed to create Xorg discoverer: failed to locate libcuda.so: pattern libcuda.so.*.*.* not found: unknown```
