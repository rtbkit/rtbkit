CXX ?= g++
CXXFLAGS := -I. -pipe -Wall -Werror -Wno-sign-compare -Woverloaded-virtual -O3 -fPIC -m64 -g -I/usr/include/eigen2
CXXLINKFLAGS = -L$(BIN)  -Wl,--rpath,$(BIN) -Wl,--rpath,$(PWD)/$(BIN) -rdynamic
CXXLIBRARYFLAGS = -shared $(CXXLINKFLAGS)
CXXEXEFLAGS =$(if $(MEMORY_ALLOC_LIBRARY),-l$(MEMORY_ALLOC_LIBRARY)) $(CXXLINKFLAGS)
CXXDEBUGFLAGS := -O0 -g

FC ?= gfortran
FFLAGS := -I. -fPIC


CAL_ENABLED ?= 0

ifeq ($(CAL_ENABLED),1)

CAL_PATH := /usr/local/amdcal/
CAL_INCLUDE_PATH := /usr/local/amdcal/include
CXXFLAGS += -I$(CAL_INCLUDE_PATH) -DJML_USE_CAL=1

endif

CUDA_ENABLED ?= 0

ifeq ($(CUDA_ENABLED),1)
include arch/cuda.mk
endif

#MEMORY_ALLOC_LIBRARY := tcmalloc


