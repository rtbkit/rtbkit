$(eval $(call nodejs_addon,v8_inheritance_module,v8_inheritance_module.cc,js))
$(eval $(call vowsjs_test,v8_inheritance_test,v8_inheritance_module))

$(eval $(call nodejs_addon,js_exception_passing_module,js_exception_passing_module.cc,js))
$(eval $(call vowsjs_test,js_exception_passing_test,js_exception_passing_module))

$(eval $(call nodejs_addon,js2json_module,js2json_module.cc,js jsoncpp))
$(eval $(call vowsjs_test,js2json_test,js2json_module))

$(eval $(call nodejs_module,testlib1,testlib1.js,js2json_module))
$(eval $(call nodejs_module,testlib2,testlib2.js,testlib1))

$(eval $(call vowsjs_test,nodejs_library_test,testlib2))

$(eval $(call test,js_call_test,arch js,boost))

$(eval $(call nodejs_addon,js_variable_arity_module,js_variable_arity_module.cc,js))
$(eval $(call vowscoffee_test,js_variable_arity_test,js_variable_arity_module))

