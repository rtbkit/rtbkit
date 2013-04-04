# RTBKIT post auction makefile

LIBRTB_POST_AUCTION_SOURCES := \
	post_auction_loop.cc

LIBRTB_POST_AUCTION_LINK := \
	agent_configuration zeromq boost_thread logger opstats crypto++ leveldb gc services banker rtb

$(eval $(call library,post_auction,$(LIBRTB_POST_AUCTION_SOURCES),$(LIBRTB_POST_AUCTION_LINK)))

# post auction runner
$(eval $(call program,post_auction_runner,post_auction services banker boost_program_options))
