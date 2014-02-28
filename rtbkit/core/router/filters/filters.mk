#------------------------------------------------------------------------------#
# filters.mk
# Rémi Attab, 25 Jul 2013
# Copyright (c) 2013 Datacratic.  All rights reserved.
#
# Router filter makefile
#
# WARNING: This makefile is currently not being because of a bug in # jml-build.
# See the parent makefile for the actual build targets.
# ------------------------------------------------------------------------------#

LIB_FILTERS_SOURCES := \
	static_filters.cc \
        creative_filters.cc

LIB_FILTERS_LINK := \
	arch utils filter_registry agent_configuration rtb

$(eval $(call library,static_filters,$(LIB_FILTERS_SOURCES),$(LIB_FILTERS_LINK)))

$(eval $(call include_sub_make,filters_test,testing,filters_test.mk))
