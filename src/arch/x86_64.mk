CAL_ENABLED ?= 0

ifeq ($(CAL_ENABLED),1)

CAL_PATH := /usr/local/amdcal/
CAL_INCLUDE_PATH := /usr/local/amdcal/include

endif

CUDA_ENABLED ?= 0

ifeq ($(CUDA_ENABLED),1)
include arch/cuda.mk
endif

#MEMORY_ALLOC_LIBRARY := tcmalloc

CXX ?= g++
CXXFLAGS := -I. -pipe -Wall -Werror -Wno-sign-compare -Woverloaded-virtual -O3 -fPIC -m64 -g -I/usr/include/eigen2 $(if $(findstring 1,$(CUDA_ENABLED)),-I$(CUDA_INCLUDE_PATH) -DJML_CUDA_ENABLED=1) $(if $(findstring 1,$(CAL_ENABLED)),-I$(CAL_INCLUDE_PATH) -DJML_CAL_ENABLED=1)
CXXLINKFLAGS := -L$(BIN)  -Wl,--rpath,$(BIN) $(if $(findstring 1,$(CUDA_ENABLED)),-L$(CUDA_LIBRARY_PATH)) -Wl,--rpath,$(CUDA_LIBRARY_PATH)
CXXLIBRARYFLAGS := -shared $(CXXLINKFLAGS)
CXXEXEFLAGS :=$(if $(MEMORY_ALLOC_LIBRARY),-l$(MEMORY_ALLOC_LIBRARY)) $(CXXLINKFLAGS)
CXXDEBUGFLAGS := -O0 -g

FC ?= gfortran
FFLAGS := -I. -fPIC

