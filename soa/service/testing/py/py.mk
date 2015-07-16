$(eval $(call python_addon,python_mongo_temp_server_wrapping,mongo_temp_server_wrapping.cc,mongo_tmp_server boost_filesystem))
$(eval $(call python_module,mongo_temp_server_wrapping,$(notdir $(wildcard $(CWD)/*.py)),python_mongo_temp_server_wrapping))

$(eval $(call python_test,mongo_temp_server_wrapping_test,mongo_temp_server_wrapping))
