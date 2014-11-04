$(eval $(call library,bid_test_utils,exchange_source.cc,bid_request rtb))

$(eval $(call library,bid_request_synth,bid_request_synth.cc,arch utils jsoncpp))
$(eval $(call test,bid_request_synth_test,bid_request_synth,boost))
$(eval $(call test,currency_test,bid_request,boost))
$(eval $(call test,filter_test,filter_registry,boost))
$(eval $(call test,bids_test,rtb,boost))

