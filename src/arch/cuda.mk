CUDA_INSTALL_PATH ?= /usr/local/cuda
CUDA_PATH := $(CUDA_INSTALL_PATH)
CUDA_INCLUDE_PATH := $(CUDA_PATH)/include
CUDA_LIBRARY_PATH := $(CUDA_PATH)/lib

NVCC       := $(CUDA_INSTALL_PATH)/bin/nvcc -D_FORTIFY_SOURCE=0 -shared -Xcompiler -fPIC,-g,-O3,-fno-access-control -arch=sm_13 --ptxas-options=-v --device-emulation

