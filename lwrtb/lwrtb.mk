# RTBKIT common makefile
PYTHON_INCLUDE_PATH  = /usr/include/python$(PYTHON_VERSION)

LIBLWRTB_SOURCES := \
	bidder.cpp 

LIBLWRTB_LINK := \
	jsoncpp bid_request bidding_agent  bid_request rtb agent_configuration

$(eval $(call library,lwrtb,$(LIBLWRTB_SOURCES),$(LIBLWRTB_LINK)))

ifeq ($(PYTHON_ENABLED),1)
LWRTB_PYTHON_SOURCES := \
	lwrtb_python.i

$(eval $(call set_compile_option,$(LWRTB_PYTHON_SOURCE),-I(PYTHON_INCLUDE_PATH)))

LWRTB_PYTHON_LINK := lwrtb

$(eval $(call library,lwrtb_python,$(LWRTB_PYTHON_SOURCES),$(LWRTB_PYTHON_LINK),_lwrtb))

$(eval $(call python_module,lwrtb,__init__.py,,lwrtb_python))


endif # python


$(eval $(call include_sub_make,testing,,lwrtb_testing.mk))
