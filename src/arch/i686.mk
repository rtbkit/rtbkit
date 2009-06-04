CAL_ENABLED ?= 0
CUDA_ENABLED ?= 0

include arch/x86_64.mk

CXX := colorccache g++
CXXFLAGS += -m32 -march=i686 -msse2

