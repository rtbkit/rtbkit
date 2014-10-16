$(eval $(call test,decision_tree_xor_test,boosting utils arch worker_task,boost))
$(eval $(call test,split_test,boosting,boost))
$(eval $(call test,decision_tree_multithreaded_test,boosting utils arch worker_task,boost))
$(eval $(call test,decision_tree_unlimited_depth_test,boosting utils arch worker_task,boost))
$(eval $(call test,glz_classifier_test,boosting utils arch worker_task,boost))
$(eval $(call test,probabilizer_test,boosting utils arch,boost))
$(eval $(call test,feature_info_test,boosting utils arch,boost))
$(eval $(call test,weighted_training_test,boosting,boost))

$(eval $(call program,dataset_nan_test,boosting utils arch boosting_tools))

ifeq ($(CUDA_ENABLED),1)
$(eval $(call test,split_cuda_test,boosting_cuda,boost))
endif # CUDA_ENABLED
