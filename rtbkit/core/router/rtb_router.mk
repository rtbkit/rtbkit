# RTBKIT router makefile

# \todo We really should include filters/filters.mk instead but
# jml-build explodes if we attempt to include a makefile before
# building a library.

LIB_FILTERS_SOURCES := \
	filters/static_filters.cc \
        filters/creative_filters.cc

LIB_FILTERS_LINK := \
	arch utils filter_registry agent_configuration rtb

$(eval $(call library,static_filters,$(LIB_FILTERS_SOURCES),$(LIB_FILTERS_LINK)))

LIBRTB_ROUTER_SOURCES := \
	augmentation_loop.cc \
	router.cc \
	router_types.cc \
	router_stack.cc \
	filter_pool.cc

LIBRTB_ROUTER_LINK := \
	rtb zeromq boost_thread logger opstats crypto++ leveldb gc services redis banker gobanker agent_configuration monitor monitor_service post_auction static_filters openrtb

$(eval $(call library,rtb_router,$(LIBRTB_ROUTER_SOURCES),$(LIBRTB_ROUTER_LINK)))

$(eval $(call program,router_runner,rtb_router boost_program_options))

$(eval $(call include_sub_make,rtb_router_testing,testing,rtb_router_testing.mk))
$(eval $(call include_sub_make,filters_test,filters/testing))
