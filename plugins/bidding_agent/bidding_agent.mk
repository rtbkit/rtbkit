# RTB simulator makefile
# Jeremy Barnes, 16 January 2010

LIBRTB_ROUTER_PROXY_SOURCES := \
	bidding_agent.cc

LIBRTB_ROUTER_PROXY_LINK := \
	ACE arch utils jsoncpp boost_thread zmq opstats bid_request services

$(eval $(call library,bidding_agent,$(LIBRTB_ROUTER_PROXY_SOURCES),$(LIBRTB_ROUTER_PROXY_LINK)))
