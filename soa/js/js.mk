# js.mk
# Jeremy Barnes, 11 May 2010
# Copyright (c) 2010 Datacratic.  All rights reserved.
#
# Support functions for javascript

LIBJS_SOURCES := \
	js_wrapped.cc \
	js_value.cc \
	js_utils.cc \
	js_registry.cc \
	js_call.cc

LIBJS_LINK := jsoncpp $(NODE_V8_LIB) arch utils types

$(eval $(call library,js,$(LIBJS_SOURCES),$(LIBJS_LINK)))

$(eval $(call include_sub_make,js_testing,testing,js_testing.mk))
