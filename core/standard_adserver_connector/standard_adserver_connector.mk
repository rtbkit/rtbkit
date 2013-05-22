# Standard Adserver Connector makefile
# Wolfgang Sourdeau, February, March 2013

# standard adserver connector
LIBSTANDARDADSERVER_SOURCES := \
	standard_adserver_connector.cc

LIBSTANDARDADSERVER_LINK := \
	adserver_connector

$(eval $(call library,standard_adserver_connector, \
	$(LIBSTANDARDADSERVER_SOURCES),	           \
	$(LIBSTANDARDADSERVER_LINK)))

# adserver connector
$(eval $(call program,standard_adserver_connector_runner, \
	opstats zmq logger boost_program_options standard_adserver_connector))
