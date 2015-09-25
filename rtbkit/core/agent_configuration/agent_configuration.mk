# RTBKIT agent configuration makefile

LIBAGENT_CONFIGURATION_SOURCES := \
	agent_config.cc \
	blacklist.cc \
	include_exclude.cc \
	agent_configuration_listener.cc \
	agent_configuration_service.cc \
	latlonrad.cc \
	fees.cc \

LIBAGENT_CONFIGURATION_LINK := \
	rtb zeromq boost_thread opstats gc services utils monitor

$(eval $(call library,agent_configuration,$(LIBAGENT_CONFIGURATION_SOURCES),$(LIBAGENT_CONFIGURATION_LINK)))

$(eval $(call program,agent_configuration_service_runner,agent_configuration boost_program_options opstats))

$(eval $(call include_sub_make,rtb_agent_configuration_testing,testing,rtb_agent_configuration_testing.mk))

