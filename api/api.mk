# RTBKIT common makefile


LIBAPI_SOURCES := \
	bidder.cpp 

LIBAPI_LINK := \
	jsoncpp bid_request bidding_agent  bid_request rtb agent_configuration

$(eval $(call library,api,$(LIBAPI_SOURCES),$(LIBAPI_LINK)))

$(eval $(call include_sub_make,testing,,api_testing.mk))
