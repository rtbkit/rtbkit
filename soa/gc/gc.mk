#------------------------------------------------------------------------------#
# gc.mk
# Rémi Attab, 01 Feb 2013
# Copyright (c) 2013 Datacratic.  All rights reserved.
#
# dasdb and rtbkit's rcu mechanism makefile.
#------------------------------------------------------------------------------#


LIBGC_SOURCES := \
	gc_lock.cc 

$(eval $(call library,gc,$(LIBGC_SOURCES),arch utils urcu))

$(eval $(call include_sub_make,gc_testing,testing,gc_testing.mk))
