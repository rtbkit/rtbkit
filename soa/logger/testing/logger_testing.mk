$(eval $(call nodejs_test,logger_js_test,logger,,,manual))
$(eval $(call nodejs_test,remote_logger_test,logger))
$(eval $(call test,remote_logger_test2,logger,boost))
$(eval $(call nodejs_test,filter_js_test,logger sync))
$(eval $(call test,json_filter_test,logger,boost manual))

$(eval $(call vowscoffee_test,logger_fs_test,logger))
$(eval $(call test,logger_deadlock_test,logger,boost manual))

$(eval $(call test,multi_output_logger_test,logger,boost))
$(eval $(call test,rotating_file_logger_test,logger,manual boost))

$(eval $(call vowscoffee_test,logger_metrics_interface_js_test,iloggermetricscpp))

#$(eval $(call test,logger_metrics_test,log_metrics mongo_tmp_server utils,manual boost))
