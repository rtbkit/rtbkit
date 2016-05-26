# analytics makefile

$(eval $(call library,analytics_endpoint,analytics_endpoint.cc,services))
$(eval $(call program,analytics_runner,analytics_endpoint boost_program_options))

$(eval $(call library,zmq_analytics,zmq_analytics.cc,zmq services rtb_router))

$(eval $(call include_sub_make,analytics_testing,testing,analytics_testing.mk))
