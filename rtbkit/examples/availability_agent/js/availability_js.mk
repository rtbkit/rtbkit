# availability_js.mk
# Rémi Attab
# Copyright (c) 2012 Recoset.  All rights reserved.
#
# Availability Node module makefile

LIBAVAILABILITY_JS_SOURCES := availability_js.cc
LIBAVAILABILITY_JS_LINK := js sigslot availability

$(eval $(call nodejs_addon,availability,$(LIBAVAILABILITY_JS_SOURCES),$(LIBAVAILABILITY_JS_LINK),services))
