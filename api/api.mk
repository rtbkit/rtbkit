# RTBKIT common makefile


LIBRTB_SOURCES := \
	bidder.cpp 

LIBRTB_LINK := \
	jsoncpp bid_request

$(eval $(call library,api,$(LIBRTB_SOURCES),$(LIBRTB_LINK)))

$(eval $(call include_sub_make,testing,,api_testing.mk))
