# RTBKIT exchange makefile

LIBRTB_EXCHANGE_SOURCES := \
	exchange_connector.cc \
	http_exchange_connector.cc \
	http_auction_handler.cc \
	ad_server_connector.cc

LIBRTB_EXCHANGE_LINK := \
	zeromq boost_thread

$(eval $(call library,exchange,$(LIBRTB_EXCHANGE_SOURCES),$(LIBRTB_EXCHANGE_LINK)))
