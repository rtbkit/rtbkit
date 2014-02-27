$(eval $(call library,pipeline,code.cc,value_description services))
$(eval $(call include_sub_make,pipeline_testing,testing,pipeline_testing.mk))
