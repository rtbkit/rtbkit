# Availability Check Agent make file.
# Rémi Attab, 31 July 2012

LIBAVAILABILITY_SOURCES := \
	availability_check.cc \
	availability_agent.cc

LIBAVAILABILITY_LINK := \
	arch utils services types gc bidding_agent rtb_router opstats zmq jsoncpp boost_program_options

$(eval $(call library,availability,$(LIBAVAILABILITY_SOURCES),$(LIBAVAILABILITY_LINK)))
$(eval $(call program,availability_runner,availability))

$(eval $(call include_sub_make,availability_js,js,availability_js.mk))
