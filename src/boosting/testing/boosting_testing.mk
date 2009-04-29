$(eval $(call test,decision_tree_xor_test,boosting,boost))
$(eval $(call test,split_test,boosting,boost))
$(eval $(call test,split_cuda_test,boosting_cuda,boost))
