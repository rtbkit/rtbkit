#------------------------------------------------------------------------------#
# rtb_augmentor.mk
# Rémi Attab, 14 Feb 2013
# Copyright (c) 2013 Datacratic.  All rights reserved.
#
# RTBKit augmentor base makefile
#------------------------------------------------------------------------------#

$(eval $(call library,augmentor_base,augmentor_base.cc,zmq rtb bid_request services))
$(eval $(call include_sub_make,augmentor_testing,testing,augmentor_testing.mk))
