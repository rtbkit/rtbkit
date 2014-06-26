$(eval $(call test,named_endpoint_test,services,boost manual))
$(eval $(call test,zmq_named_pub_sub_test,services,boost manual))
$(eval $(call test,zmq_endpoint_test,services,boost manual))
$(eval $(call test,message_channel_test,services,boost))
$(eval $(call test,rest_service_endpoint_test,services,boost))
$(eval $(call test,multiple_service_test,services,boost manual))

$(eval $(call test,zookeeper_test,cloud,boost manual))
$(eval $(call test,aws_test,cloud,boost))

$(eval $(call test,redis_async_test,redis,boost))
$(eval $(call test,redis_commands_test,redis,boost))

$(eval $(call nodejs_test,opstats_js_test,opstats,,,manual))

$(eval $(call test,statsd_connector_test,opstats,boost  manual))
$(eval $(call test,carbon_connector_test,opstats endpoint,boost manual))

$(eval $(call test,endpoint_unit_test,endpoint,boost))
$(eval $(call test,test_active_endpoint_nothing_listening,endpoint,boost manual))
$(eval $(call test,test_active_endpoint_not_responding,endpoint,boost manual))
$(eval $(call test,test_endpoint_ping_pong,endpoint,boost manual))
$(eval $(call test,test_endpoint_connection_speed,endpoint,boost manual))
$(eval $(call test,test_endpoint_accept_speed,endpoint,boost))
$(eval $(call test,endpoint_periodic_test,endpoint,boost))
$(eval $(call test,endpoint_closed_connection_test,endpoint,boost))
$(eval $(call test,http_long_header_test,endpoint,boost manual))
$(eval $(call test,http_header_test,endpoint,boost manual))
$(eval $(call test,service_proxies_test,endpoint,boost manual))

$(eval $(call test,message_loop_test,services,boost))

$(eval $(call program,runner_test_helper,utils))
$(eval $(call test,runner_test,services,boost))
$(eval $(call test,runner_stress_test,services,boost))
$(TESTS)/runner_test $(TESTS)/runner_stress_test: $(BIN)/runner_test_helper
$(eval $(call test,sink_test,services,boost))

$(eval $(call library,tcpsockets,tcpsockets.cc,services))
$(eval $(call test,zmq_tcp_bench,tcpsockets services,boost manual timed))
$(eval $(call test,nprobe_test,services,boost manual))

$(eval $(call library,test_services,test_http_services.cc,services))

$(eval $(call test,http_client_test,services test_services,boost))
$(eval $(call test,http_client_bench,boost_program_options services test_services,boost manual))

$(eval $(call test,logs_test,services,boost))

$(eval $(call test,sns_mock_test,cloud services,boost))
$(eval $(call test,zmq_message_loop_test,services,boost))
