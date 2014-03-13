# Makefile for boosting functions
# Jeremy Barnes, 1 April 2006
# Copyright (c) 2006 Jeremy Barnes.  All rights reserved.

LIBBOOSTING_SOURCES := \
	boosted_stumps.cc \
        classifier.cc \
        data_aliases.cc \
        decoded_classifier.cc \
        decision_tree.cc \
        null_feature_space.cc \
        decoder.cc \
        dense_features.cc \
        evaluation.cc \
        feature_info.cc \
        feature_set.cc \
        feature_space.cc \
        glz_classifier.cc \
        naive_bayes.cc \
        null_classifier.cc \
        null_decoder.cc \
        probabilizer.cc \
        sparse_features.cc \
        stump.cc \
        training_data.cc \
        training_index.cc \
        training_index_entry.cc \
        weighted_training.cc \
        transformed_classifier.cc \
        stump_training.cc \
        config_options.cc \
        stump_regress.cc \
        boosted_stumps_generator.cc \
        bagging_generator.cc \
        boosting_generator.cc \
        naive_bayes_generator.cc \
        decision_tree_generator.cc \
        feature_transformer.cc \
        glz_classifier_generator.cc \
        classifier_generator.cc \
        stump_generator.cc \
        binary_symmetric.cc \
        early_stopping_generator.cc \
        stump_training_bin.cc \
        feature_transform.cc \
        transform_list.cc \
        committee.cc \
        boosting_training.cc \
        null_classifier_generator.cc \
	tree.cc \
	split.cc \
	training_index_iterators.cc \
	feature.cc \
	bit_compressed_index.cc \
	label.cc \
	buckets.cc

LIBBOOSTING_LINK :=	utils db algebra arch judy ACE boost_regex boost_thread worker_task

#$(eval $(call set_compile_option,perceptron_generator.cc perceptron.cc,-ffast-math))

$(eval $(call library,boosting,$(LIBBOOSTING_SOURCES),$(LIBBOOSTING_LINK)))

ifeq ($(CUDA_ENABLED),1)

LIBBOOSTING_CUDA_SOURCES := stump_training_cuda.cu stump_training_cuda_host.cc backprop_cuda.cu
LIBBOOSTING_CUDA_LINK := boosting arch_cuda cudart_ocelot

$(eval $(call library,boosting_cuda,$(LIBBOOSTING_CUDA_SOURCES),$(LIBBOOSTING_CUDA_LINK)))

endif # CUDA_ENABLED

$(eval $(call include_sub_make,boosting_tools,tools))
$(eval $(call include_sub_make,boosting_testing,testing))
