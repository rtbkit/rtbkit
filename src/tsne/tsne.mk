# Makefile for tsne functionality
# Jeremy Barnes, 16 January 2010
# Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

LIBTSNE_SOURCES := \
        tsne.cc \

LIBTSNE_LINK :=	utils algebra arch boost_thread-mt stats

$(eval $(call library,tsne,$(LIBTSNE_SOURCES),$(LIBTSNE_LINK)))

ifeq ($(CUDA_ENABLED),1)

LIBTSNE_CUDA_SOURCES := backprop_cuda.cu
LIBTSNE_CUDA_LINK := tsne arch_cuda cudart_ocelot

$(eval $(call library,tsne_cuda,$(LIBTSNE_CUDA_SOURCES),$(LIBTSNE_CUDA_LINK)))

endif # CUDA_ENABLED

$(eval $(call include_sub_make,tsne_testing,testing))
