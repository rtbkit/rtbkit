#------------------------------------------------------------------------------#
# examples.mk
# Rémi Attab, 14 Feb 2013
# Copyright (c) 2013 Datacratic.  All rights reserved.
#
# Makefile for various RTBkit examples. 
#------------------------------------------------------------------------------#

$(eval $(call program,augmentor_ex,augmentor_base rtb bid_request agent_configuration boost_program_options))
$(eval $(call program,data_logger_runner,data_logger data_logger boost_program_options services))
$(eval $(call program,bidding_agent_console,bidding_agent rtb_router boost_program_options services))


RTBKIT_INTEGRATION_TEST_LINK := \
	rtb_router bidding_agent integration_test_utils monitor monitor_service
$(eval $(call program,rtbkit_integration_test,$(RTBKIT_INTEGRATION_TEST_LINK)))
