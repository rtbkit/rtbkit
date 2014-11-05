# date.mk
# Jeremy Barnes, 2 November 2011
# Copyright (c) 2011 Datacratic.  All rights reserved.
#
# Date library for recoset.

LIBTYPES_SOURCES := \
	date.cc \
	localdate.cc \
	string.cc \
	id.cc \
	url.cc \
	periodic_utils.cc \
	csiphash.c

LIBTYPES_LINK := \
	boost_regex boost_date_time jsoncpp ACE db googleurl cityhash utils

$(eval $(call set_compile_option,localdate.cc,-DLIB=\"$(LIB)\"))

ifneq ($(PREMAKE),1)
$(LIB)/libtypes.so: $(LIB)/date_timezone_spec.csv

$(LIB)/date_timezone_spec.csv: $(CWD)/date_timezone_spec.csv
	@echo "           $(COLOR_CYAN)[COPY]$(COLOR_RESET) $< -> $@"
	@/bin/cp -f $< $@
endif

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
