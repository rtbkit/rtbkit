# RTBKIT makefile
#
#
LIBRTBKIT_SOURCES := \
	rtbkit.cc

$(eval $(call library,rtbkit,$(LIBRTBKIT_SOURCES)))


$(eval $(call include_sub_make,openrtb))
$(eval $(call include_sub_make,common))
$(eval $(call include_sub_make,core))
$(eval $(call include_sub_make,plugins))
$(eval $(call include_sub_make,js))
$(eval $(call include_sub_make,testing))
$(eval $(call include_sub_make,examples))

