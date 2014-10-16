#------------------------------------------------------------------------------#
# augmentor_testing.mk
# Rémi Attab, 10 May 2013
# Copyright (c) 2013 Datacratic.  All rights reserved.
#
# Makefile for the augmentor base class tests.
#------------------------------------------------------------------------------#

$(eval $(call test,augmentor_stress_test,augmentor_base bid_request,boost manual))
$(eval $(call test,redis_augmentor_test,augmentor_base bid_request bidding_agent,boost))


