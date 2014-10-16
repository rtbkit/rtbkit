# Monitor testing makefile
# Wolfgang Sourdeau, January 2013

# MonitorEndpoint
$(eval $(call test,monitor_endpoint_test,monitor_service,boost))

# MonitorClient
$(eval $(call test,monitor_client_test,monitor,boost))

# Integration test between all Monitor components
$(eval $(call test,monitor_behaviour_test,monitor monitor_service,boost manual))

monitor_tests: monitor_test monitor_client_test monitor_behaviour_test
