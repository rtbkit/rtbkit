$(eval $(call library,bid_test_utils,exchange_source.cc,bid_request))

$(eval $(call library,bid_request_synth,bid_request_synth.cc,arch utils jsoncpp))
$(eval $(call test,bid_request_synth_test,bid_request_synth,boost))
