#------------------------------------------------------------------------------#
# testing.mk
# Rémi Attab, 15 Feb 2013
# Copyright (c) 2013 Datacratic.  All rights reserved.
#
# Tests and test utilities for RTBKit
#------------------------------------------------------------------------------#

$(eval $(call vowscoffee_test,bid_request_js_test,bid_request))
$(eval $(call test,agent_configuration_test,rtb_router bidding_agent,boost))
$(eval $(call test,augmentation_list_test,rtb,boost))
$(eval $(call test,historical_bid_request_test,bid_request,boost))

$(eval $(call library,integration_test_utils,generic_exchange_connector.cc mock_exchange.cc,rtb_router exchange))

$(eval $(call program,mock_exchange_runner,integration_test_utils boost_program_options utils))
$(eval $(call program,json_feeder,curlpp boost_program_options utils))
$(eval $(call program,json_listener,boost_program_options services utils))
