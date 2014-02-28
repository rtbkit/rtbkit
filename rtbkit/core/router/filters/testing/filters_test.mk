#------------------------------------------------------------------------------#
# filters_test.mk
# Rémi Attab, 15 Aug 2013
# Copyright (c) 2013 Datacratic.  All rights reserved.
#
# Makefile for the various filter tests.
#------------------------------------------------------------------------------#

$(eval $(call test,generic_filters_test,static_filters,boost))
$(eval $(call test,static_filters_test,static_filters,boost))
$(eval $(call test,creative_filters_test,static_filters,boost))


