$(eval $(call test,signal_test,sigslot,boost valgrind))
$(eval $(call test,slot_test,sigslot,boost valgrind))

$(eval $(call nodejs_addon,signal_slot_test_module,signal_slot_test_module.cc,,sigslot))
$(eval $(call vowsjs_test,signal_slot_js_test,signal_slot_test_module sync))
