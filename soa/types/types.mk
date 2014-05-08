# date.mk
# Jeremy Barnes, 2 November 2011
# Copyright (c) 2011 Datacratic.  All rights reserved.
#
# Date library for recoset.

BOOST_TIMEZONES_DIR := $(shell pwd)/$(CWD)

LIBTYPES_SOURCES := \
	date.cc \
	localdate.cc \
	string.cc \
	id.cc \
	url.cc \
	periodic_utils.cc \

LIBTYPES_LINK := \
	boost_regex boost_date_time jsoncpp ACE db googleurl cityhash utils

$(eval $(call set_compile_option,localdate.cc,-DBOOST_TIMEZONES_DIR=\"$(BOOST_TIMEZONES_DIR)\"))
$(eval $(call library,types,$(LIBTYPES_SOURCES),$(LIBTYPES_LINK)))

LIBVALUE_DESCRIPTION_SOURCES := \
	value_description.cc \
	json_parsing.cc \
	json_printing.cc \
	periodic_utils_value_descriptions.cc

LIBVALUE_DESCRIPTION_LINK := \
	arch types

$(eval $(call library,value_description,$(LIBVALUE_DESCRIPTION_SOURCES),$(LIBVALUE_DESCRIPTION_LINK)))


$(eval $(call include_sub_make,types_testing,testing,types_testing.mk))
