# RTBKIT common makefile


LIBRTB_SOURCES := \
	bidder.cc 

LIBRTB_LINK := \
	jsoncpp bid_request

$(eval $(call library,rtb,$(LIBRTB_SOURCES),$(LIBRTB_LINK)))

$(eval $(call include_sub_make,testing,,api_testing.mk))
