# analytics makefile

$(eval $(call library,analytics,analytics_endpoint.cc,services))
$(eval $(call program,analytics_runner,analytics boost_program_options))

$(eval $(call include_sub_make,analytics_testing,testing,analytics_testing.mk))
