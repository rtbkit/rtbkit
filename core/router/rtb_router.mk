# RTBKIT router makefile

LIBRTB_ROUTER_SOURCES := \
	augmentation_loop.cc \
	router.cc \
	router_types.cc \
	router_stack.cc

LIBRTB_ROUTER_LINK := \
	rtb zeromq boost_thread logger opstats crypto++ leveldb gc services redis banker agent_configuration monitor monitor_service post_auction

$(eval $(call library,rtb_router,$(LIBRTB_ROUTER_SOURCES),$(LIBRTB_ROUTER_LINK)))

$(eval $(call library,router_runner,router_runner.cc,rtb_router boost_program_options))

$(eval $(call include_sub_make,rtb_router_testing,testing,rtb_router_testing.mk))
