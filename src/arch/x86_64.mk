include arch/cuda.mk

CAL_PATH := /usr/local/amdcal/
CAL_INCLUDE_PATH := /usr/local/amdcal/include

CAL_ENABLED := 0

#MEMORY_ALLOC_LIBRARY := tcmalloc

CXX := colorccache g++
CXXFLAGS := -I. -pipe -Wall -Werror -Wno-sign-compare -Woverloaded-virtual -O3 -fPIC -m64 -g -I/usr/include/eigen2 -I$(CUDA_INCLUDE_PATH) $(if $(CAL_ENABLED),-I$(CAL_INCLUDE_PATH))
CXXLINKFLAGS := -shared -L$(BIN) -L$(CUDA_LIBRARY_PATH) -Wl,--rpath,$(BIN),--rpath,$(CUDA_LIBRARY_PATH)
CXXEXEFLAGS :=	$(if $(MEMORY_ALLOC_LIBRARY),-l$(MEMORY_ALLOC_LIBRARY)) -L$(BIN) -L$(CUDA_LIBRARY_PATH) -Wl,--rpath,$(BIN),--rpath,$(CUDA_LIBRARY_PATH)
CXXDEBUGFLAGS := -O0 -g

FC := colorccache gfortran
FFLAGS := -I. -fPIC

