CXX ?= g++
CXXFLAGS ?= $(INCLUDE) -pipe -Wall -Werror -Wno-sign-compare -Woverloaded-virtual -fPIC -m64 -ggdb -fno-omit-frame-pointer $(foreach dir,$(LOCAL_INCLUDE_DIR),-I$(dir)) -std=c++0x -Wno-deprecated-declarations
CXXLINKFLAGS = -rdynamic $(foreach DIR,$(PWD)/$(BIN) $(PWD)/$(LIB) $(LOCAL_LIB_DIR),-L$(DIR) -Wl,--rpath-link,$(DIR)) -Wl,--rpath,\$$ORIGIN/../bin -Wl,--rpath,\$$ORIGIN/../lib -Wl,--copy-dt-needed-entries -Wl,--no-as-needed
CXXLIBRARYFLAGS = -shared $(CXXLINKFLAGS) -lpthread
CXXEXEFLAGS =$(CXXLINKFLAGS) -lpthread
CXXEXEPOSTFLAGS := $(if $(MEMORY_ALLOC_LIBRARY),-l$(MEMORY_ALLOC_LIBRARY))
CXXNODEBUGFLAGS := -O3 -DBOOST_DISABLE_ASSERTS -DNDEBUG 
CXXDEBUGFLAGS := -O1 -g3 -UBOOST_DISABLE_ASSERTS -UNDEBUG

CC ?= gcc
CFLAGS ?= $(INCLUDE) -pipe -Wall -Werror -Wno-sign-compare -O3 -fPIC -m64 -g -DNDEBUG -fno-omit-frame-pointer
CDEDEBUGFLAGS := -O0 -g -UNDEBUG

FC ?= gfortran
FFLAGS ?= -I. -fPIC

PYTHON_ENABLED ?= 1

VALGRIND ?= valgrind
VALGRINDFLAGS ?= --error-exitcode=1 --leak-check=full

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


