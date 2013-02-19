# Monitor testing makefile
# Wolfgang Sourdeau, January 2013

# Monitor
$(eval $(call test,monitor_test,monitor_service,boost))
$(eval $(call test,monitor_provider_proxy_test,monitor_service,boost))
#$(eval $(call test,monitor_behaviour_test,monitor monitor_service,boost))

# Monitor Provider
$(eval $(call test,monitor_provider_test,monitor,boost))

# Monitor Proxy
$(eval $(call test,monitor_proxy_test,monitor,boost))

monitor_tests: monitor_test monitor_provider_test monitor_provider_proxy_test\
	 monitor_proxy_test monitor_behaviour_test
