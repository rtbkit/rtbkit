# RTBKIT adserver makefile

LIBADSERVERCONNECTOR_SOURCES := \
	adserver_connector.cc \
	http_adserver_connector.cc

LIBADSERVERCONNECTOR_LINK := \
	zeromq boost_thread utils endpoint services rtb

$(eval $(call library,adserver_connector, \
	$(LIBADSERVERCONNECTOR_SOURCES),  \
	$(LIBADSERVERCONNECTOR_LINK)))

$(eval $(call library,standard_adserver_connector,standard_adserver_connector.cc,adserver_connector))
$(eval $(call program,standard_adserver_connector_runner,opstats zmq logger boost_program_options standard_adserver_connector))
