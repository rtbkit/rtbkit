# RTBKIT filter makefile

LIB_FILTERS_LINK := \
	arch utils filter_registry agent_configuration rtb

$(eval $(call library,custom_filter,custom_filter.cc,$(LIB_FILTERS_LINK)))

$(eval $(call include_sub_make,filter_testing,testing,filter_testing.mk))
