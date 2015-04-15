# RTBKIT post auction makefile

LIB_POST_AUCTION_SOURCES := \
        simple_event_matcher.cc \
	sharded_event_matcher.cc \
	events.cc \
	finished_info.cc \
	post_auction_service.cc

LIB_POST_AUCTION_LINK := \
	agent_configuration zeromq boost_thread logger opstats leveldb services banker gobanker rtb

$(eval $(call library,post_auction,$(LIB_POST_AUCTION_SOURCES),$(LIB_POST_AUCTION_LINK)))


# post auction runner
$(eval $(call program,post_auction_runner,post_auction services banker gobanker boost_program_options))

$(eval $(call include_sub_make,post_auction_testing,testing,testing.mk))
