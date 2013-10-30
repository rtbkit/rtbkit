# RTBKIT common makefile
PYTHON_INCLUDE_PATH  = /usr/include/python$(PYTHON_VERSION)

LIBAPI_SOURCES := \
	bidder.cpp 

LIBAPI_LINK := \
	jsoncpp bid_request bidding_agent  bid_request rtb agent_configuration

$(eval $(call library,api,$(LIBAPI_SOURCES),$(LIBAPI_LINK)))

ifeq ($(PYTHON_ENABLED),1)
LWRTB_PYTHON_SOURCE := \
	lwrtb_python.i

$(eval $(call set_compile_option,$(LWRTB_PYTHON_SOURCE),-I(PYTHON_INCLUDE_PATH)))
endif # python


$(eval $(call include_sub_make,testing,,api_testing.mk))
