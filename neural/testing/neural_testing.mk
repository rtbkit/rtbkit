# Makefile for neural network testing
# Jeremy Barnes, 2 November 2009
# Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

$(eval $(call test,parameters_test,neural,boost))
$(eval $(call test,dense_layer_test,neural utils arch db worker_task,boost))
$(eval $(call test,layer_stack_test,neural utils arch db worker_task,boost))
$(eval $(call test,discriminative_trainer_test,neural,boost))
$(eval $(call test,twoway_layer_test,neural utils arch db worker_task,boost manual))
$(eval $(call test,perceptron_test,neural utils boosting worker_task,boost manual))
$(eval $(call test,output_encoder_test,neural,boost))

