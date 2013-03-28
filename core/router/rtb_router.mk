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

ROUTER_LOGGER_SOURCES := \
	router_logger.cc

ROUTER_LOGGER_LINK := \
	logger \
	monitor

$(eval $(call library,router_logger,$(ROUTER_LOGGER_SOURCES),$(ROUTER_LOGGER_LINK)))
$(eval $(call program,router_logger_runner,router_logger boost_program_options opstats bid_request))

$(eval $(call include_sub_make,rtb_router_testing,testing,rtb_router_testing.mk))
