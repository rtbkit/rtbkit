# RTBKIT adserver makefile

LIBADSERVERCONNECTOR_SOURCES := \
	adserver_connector.cc \
	http_adserver_connector.cc

LIBADSERVERCONNECTOR_LINK := \
	zeromq boost_thread utils endpoint services rtb

$(eval $(call library,adserver_connector, \
	$(LIBADSERVERCONNECTOR_SOURCES),  \
	$(LIBADSERVERCONNECTOR_LINK)))
