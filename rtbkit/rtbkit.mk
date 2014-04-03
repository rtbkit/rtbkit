# RTBKIT makefile
#
#
LIBRTBKIT_SOURCES := \
	rtbkit.cc

LIBRTBKIT_LINK :=                          \
        bidding_agent bid_request          \
        boost_program_options data_logger  \
        exchange rtb_router                \
        services standard_adserver

$(eval $(call library,rtbkit,$(LIBRTBKIT_SOURCES),$(LIBRTBKIT_LINK)))


$(eval $(call include_sub_make,openrtb))
$(eval $(call include_sub_make,common))
$(eval $(call include_sub_make,core))
$(eval $(call include_sub_make,plugins))
$(eval $(call include_sub_make,js))
$(eval $(call include_sub_make,testing))
$(eval $(call include_sub_make,examples))

