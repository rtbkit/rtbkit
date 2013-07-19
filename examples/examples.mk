#------------------------------------------------------------------------------#
# examples.mk
# Rémi Attab, 14 Feb 2013
# Copyright (c) 2013 Datacratic.  All rights reserved.
#
# Makefile for various RTBkit examples. 
#------------------------------------------------------------------------------#

$(eval $(call library,augmentor_ex,augmentor_ex.cc,augmentor_base rtb bid_request agent_configuration))
$(eval $(call program,augmentor_ex_runner,augmentor_ex boost_program_options))
$(eval $(call program,data_logger_ex,data_logger data_logger boost_program_options services))
$(eval $(call program,bidding_agent_console,bidding_agent rtb_router boost_program_options services))
$(eval $(call program,bidding_agent_ex,bidding_agent rtb_router boost_program_options services))
$(eval $(call program,ad_server_connector_ex,adserver_connector exchange rtb_router boost_program_options services))
$(eval $(call program,router_ex,adserver_connector rubicon_exchange fbx_exchange openrtb_exchange exchange router_runner boost_program_options services))
$(eval $(call program,bid_request_endpoint,rubicon_exchange openrtb_exchange fbx_exchange exchange rtb_router bidding_agent boost_program_options services))
$(eval $(call program,adserver_endpoint,standard_adserver_connector data_logger rtb_router bidding_agent boost_program_options services))
$(eval $(call program,integration_endpoints,rubicon_exchange openrtb_exchange fbx_exchange exchange standard_adserver_connector data_logger rtb_router bidding_agent boost_program_options services))

RTBKIT_INTEGRATION_TEST_LINK := \
	rtb_router bidding_agent integration_test_utils monitor monitor_service augmentor_ex adserver_connector

$(eval $(call test,rtbkit_integration_test,$(RTBKIT_INTEGRATION_TEST_LINK),boost))


$(eval $(call program,standalone_bidder_ex,augmentor_base rtb bid_request agent_configuration boost_program_options rtb_router adserver_connector exchange))
