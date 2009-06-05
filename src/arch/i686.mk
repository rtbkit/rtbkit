CAL_ENABLED ?= 0
CUDA_ENABLED ?= 0

include arch/x86_64.mk

CXX := g++
CXXFLAGS := $(filter-out -Werror,$(CXXFLAGS)) -m32 -march=i686 -msse2 -D_REENTRANT
FC := gfortran
