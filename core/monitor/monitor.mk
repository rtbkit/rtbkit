# RTBKIT monitor makefile

LIBRESTMULTIPROXY_SOURCES := \
	rest_multi_proxy.cc

LIBRESTMULTIPROXY_LINK := \
	services

$(eval $(call library,restmultiproxy,$(LIBRESTMULTIPROXY_SOURCES),$(LIBRESTMULTIPROXY_LINK)))

LIBMONITOR_SOURCES := \
	monitor_provider.cc \
	monitor_proxy.cc

LIBMONITOR_LINK := \
	rtb services

$(eval $(call library,monitor,$(LIBMONITOR_SOURCES),$(LIBMONITOR_LINK)))

LIBMONITORSERVICE_SOURCES := \
	monitor.cc \
	monitor_provider_proxy.cc

LIBMONITORSERVICE_LINK := \
	services \
	restmultiproxy

$(eval $(call library,monitor_service,$(LIBMONITORSERVICE_SOURCES),$(LIBMONITORSERVICE_LINK)))

$(eval $(call program,monitor_service_runner,monitor_service boost_program_options))

$(eval $(call include_sub_make,monitor_testing,testing,monitor_testing.mk))
