# RTBKIT post auction makefile

LIBRTB_POST_AUCTION_SOURCES := \
	post_auction_loop.cc

LIBRTB_POST_AUCTION_LINK := \
	zeromq boost_thread logger opstats crypto++ leveldb gc services banker

$(eval $(call library,post_auction,$(LIBRTB_POST_AUCTION_SOURCES),$(LIBRTB_POST_AUCTION_LINK)))
