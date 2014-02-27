# sigslot.mk
# Jeremy Barnes, 1 February 2011
# Copyright (c) 2011 Datacratic.  All rights reserved.
#
# Signal/slot library for Datacratic.

LIBSIGSLOT_SOURCES := \
	slot.cc \
	signal.cc \
	slot_js.cc

LIBSIGSLOT_LINK := \
	js arch

$(eval $(call library,sigslot,$(LIBSIGSLOT_SOURCES),$(LIBSIGSLOT_LINK)))

LIBSIGSLOT_JS_SOURCES := \
	slot_node.cc

LIBSIGSLOT_JS_LINK := \
	sigslot

$(eval $(call nodejs_addon,sigslot,$(LIBSIGSLOT_JS_SOURCES),$(LIBSIGSLOT_JS_LINK)))

$(eval $(call include_sub_make,sigslot_testing,testing,sigslot_testing.mk))
