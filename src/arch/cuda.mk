CUDA_INSTALL_PATH ?= /usr/local/cuda
CUDA_PATH := $(CUDA_INSTALL_PATH)
CUDA_INCLUDE_PATH := $(CUDA_PATH)/include
CUDA_LIBRARY_PATH := $(CUDA_PATH)/lib

# Set to 1 to emulate the CUDA device
CUDA_DEVICE_EMULATION ?= 0

NVCC_EMULATION       := -Xcompiler -fno-access-control --device-emulation -Xcudafe --rtti,--exceptions,--g++

NVCC       := $(CUDA_INSTALL_PATH)/bin/nvcc -D_FORTIFY_SOURCE=0 -shared -Xcompiler -fPIC,-g,-O3 -arch=sm_13 --ptxas-options=-v $(if $(findstring x1x,x$(CUDA_DEVICE_EMULATION)x),$(NVCC_EMULATION))

CXXFLAGS += -DJML_USE_CUDA=1 -I$(CUDA_INCLUDE_PATH)
CXXLIBRARYFLAGS += -L$(CUDA_LIBRARY_PATH) -Wl,--rpath,$(CUDA_LIBRARY_PATH) -L/usr/local/lib
CXXEXEPOSTFLAGS += -lOcelotIr -lOcelotParser -lOcelotExecutive -lOcelotTrace -lOcelotAnalysis -lhydrazine
