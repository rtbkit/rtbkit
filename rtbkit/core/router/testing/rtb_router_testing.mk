# RTB router testing makefile
# Jeremy Barnes, 13 January 2012

#$(eval $(call program,rtb_router_test,rtb_router boost_unit_test_framework))

$(eval $(call nodejs_test,rtb_router_unit_test,rtb sync))
$(eval $(call nodejs_test,rtb_new_format_test,bid_request sync_utils))
#$(eval $(call test,rtb_router_leak_test,rtb_router rtbsim,boost valgrind))
$(eval $(call test,pending_list_test,types,boost))
#$(eval $(call test,router_banker_test,rtb_router dataflow bidding_agent,boost))
#$(eval $(call test,augmentation_test,rtb_router bid_request augmentor_base,boost))
