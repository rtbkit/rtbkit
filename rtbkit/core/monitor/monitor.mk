# RTBKIT monitor makefile

LIBMONITOR_SOURCES := \
	monitor_client.cc \
	monitor_provider.cc

LIBMONITOR_LINK := \
	rtb services

$(eval $(call library,monitor,$(LIBMONITOR_SOURCES),$(LIBMONITOR_LINK)))

LIBMONITORSERVICE_SOURCES := \
	monitor_endpoint.cc

LIBMONITORSERVICE_LINK := \
	services \
	rtb

$(eval $(call library,monitor_service,$(LIBMONITORSERVICE_SOURCES),$(LIBMONITORSERVICE_LINK)))

$(eval $(call program,monitor_service_runner,monitor_service boost_program_options))

$(eval $(call include_sub_make,monitor_testing,testing,monitor_testing.mk))
