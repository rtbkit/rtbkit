$(eval $(call test,parse_context_test,utils,boost))
$(eval $(call test,configuration_test,utils,boost))
$(eval $(call test,environment_test,utils,boost))
$(eval $(call test,compact_vector_test,arch,boost))
$(eval $(call test,circular_buffer_test,arch,boost))
$(eval $(call test,lightweight_hash_test,arch,boost))

$(eval $(call test,worker_task_test,worker_task boost_thread-mt,boost))
