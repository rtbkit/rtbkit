$(eval $(call test,decision_tree_xor_test,boosting,boost))
$(eval $(call test,split_test,boosting,boost))
$(eval $(call test,decision_tree_multithreaded_test,boosting,boost))
$(eval $(call test,decision_tree_unlimited_depth_test,boosting,boost))

ifeq ($(CUDA_ENABLED),1)
$(eval $(call test,split_cuda_test,boosting_cuda,boost))
endif # CUDA_ENABLED