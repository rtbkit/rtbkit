# service_js.mk
# Rémi Attab
# Copyright (c) 2012 Datacratic.  All rights reserved.
#
# Frequency

$(eval $(call nodejs_addon,opstats,opstats_js.cc,opstats js))
$(eval $(call nodejs_addon,s3,s3_js.cc,js cloud))

LIBSERVICES_JS_SOURCES := \
	service_js.cc \
	service_base_js.cc

LIBSERVICES_JS_LINK := \
	js sigslot services

$(eval $(call nodejs_addon,services,$(LIBSERVICES_JS_SOURCES),$(LIBSERVICES_JS_LINK), opstats))

