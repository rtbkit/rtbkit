CXX ?= g++
CXXFLAGS ?= $(INCLUDE) -pipe -Wall -Werror -Wno-sign-compare -Woverloaded-virtual -O3 -fPIC -m64 -g -DBOOST_DISABLE_ASSERTS -DNDEBUG -fno-omit-frame-pointer -I$(LOCAL_INCLUDE_DIR)
CXXLINKFLAGS = -L$(BIN)  -Wl,--rpath,$(BIN) -Wl,--rpath,$(PWD)/$(BIN) -rdynamic $(foreach DIR,$(LOCAL_LIB_DIR),-L$(DIR) -Wl,--rpath,$(DIR))
CXXLIBRARYFLAGS = -shared $(CXXLINKFLAGS) -lpthread
CXXEXEFLAGS =$(CXXLINKFLAGS) -lpthread
CXXEXEPOSTFLAGS := $(if $(MEMORY_ALLOC_LIBRARY),-l$(MEMORY_ALLOC_LIBRARY))
CXXDEBUGFLAGS := -O0 -g -UBOOST_DISABLE_ASSERTS -UNDEBUG

CC ?= gcc
CFLAGS ?= $(INCLUDE) -pipe -Wall -Werror -Wno-sign-compare -O3 -fPIC -m64 -g -DNDEBUG -fno-omit-frame-pointer
CDEDEBUGFLAGS := -O0 -g -UNDEBUG

FC ?= gfortran
FFLAGS ?= -I. -fPIC

PYTHON_ENABLED ?= 1

VALGRIND ?= valgrind
VALGRINDFLAGS ?= --error-exitcode=1 --leak-check=full

ifeq ($(PYTHON_ENABLED),1)

PYTHON_INCLUDE_PATH ?= /usr/include/python2.6/
PYTHON ?= python2.6
PYTHONPATH ?= $(BIN)

endif

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


