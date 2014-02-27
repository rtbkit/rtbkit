# Makefile for neural functions
# Jeremy Barnes, 1 April 2006
# Copyright (c) 2006 Jeremy Barnes.  All rights reserved.

LIBNEURAL_SOURCES := \
        perceptron.cc \
	perceptron_defs.cc \
	layer.cc \
	dense_layer.cc \
	perceptron_generator.cc \
	parameters.cc \
	transfer_function.cc \
	layer_stack.cc \
	discriminative_trainer.cc \
	auto_encoder.cc \
	auto_encoder_stack.cc \
	twoway_layer.cc \
	auto_encoder_trainer.cc \
	reverse_layer_adaptor.cc \
	reconstruct_layer_adaptor.cc \
	output_encoder.cc

LIBNEURAL_LINK :=	utils db algebra arch judy ACE boost_regex boost_thread boosting stats worker_task

$(eval $(call library,neural,$(LIBNEURAL_SOURCES),$(LIBNEURAL_LINK)))

ifeq ($(CUDA_ENABLED),1)

LIBNEURAL_CUDA_SOURCES := backprop_cuda.cu
LIBNEURAL_CUDA_LINK := neural arch_cuda cudart_ocelot

$(eval $(call library,neural_cuda,$(LIBNEURAL_CUDA_SOURCES),$(LIBNEURAL_CUDA_LINK)))

endif # CUDA_ENABLED

$(eval $(call include_sub_make,neural_testing,testing))
