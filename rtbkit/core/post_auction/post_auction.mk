# RTBKIT post auction makefile

LIBRTB_POST_AUCTION_SOURCES := \
	post_auction_loop.cc

LIBRTB_POST_AUCTION_LINK := \
	agent_configuration zeromq boost_thread logger opstats crypto++ leveldb services banker rtb

$(eval $(call library,post_auction,$(LIBRTB_POST_AUCTION_SOURCES),$(LIBRTB_POST_AUCTION_LINK)))



LIB_POST_AUCTION_2_SOURCES := \
        event_matcher.cc \
	events.cc \
	finished_info.cc \
	post_auction_service.cc \
	submission_info.cc

LIB_POST_AUCTION_2_LINK := \
	agent_configuration zeromq boost_thread logger opstats leveldb services banker rtb

$(eval $(call library,post_auction_2,$(LIB_POST_AUCTION_2_SOURCES),$(LIB_POST_AUCTION_2_LINK)))


# post auction runner
$(eval $(call program,post_auction_runner,post_auction services banker boost_program_options))
