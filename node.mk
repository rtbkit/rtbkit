ifeq ($(NODEJS_ENABLED),1)

NODE ?= LD_PRELOAD=$(BIN)/libexception_hook.so node
VOWS ?= /usr/local/bin/vows
NODE_PATH := $(if $(NODE_PATH),$(NODE_PATH):)$(BIN)
NODE_TEST_DEPS ?= $(BIN)/libnode_exception_tracing.so
VOWS_TEST_DEPS ?= $(NODE_TEST_DEPS)

# add a node.js addon
# $(1): name of the addon
# $(2): source files to include in the addon
# $(3): libraries to link with

define nodejs_addon
$$(eval $$(call library,$(1),$(2),node_exception_tracing $(3),$(1),.node,[NODEJS]))

nodejs_addons: $$(LIB_$(1)_DEPS)
endef

# node test case

# $(1) name of the test (the javascript file that contains the test case)
# $(2) node.js modules on which it depends
# $(3) options to the node executable
# $(4) test name
# $(5) test options

define nodejs_test
$$(if $(trace),$$(warning called nodejs_test "$(1)" "$(2)" "$(3)" "$(4)" "$(5)"))

TEST_$(1)_COMMAND := rm -f $(TESTS)/$(1).{passed,failed} && ((set -o pipefail && NODE_PATH=$(NODE_PATH) $(NODE) $(3) $(CWD)/$(1).js > $(TESTS)/$(1).running 2>&1 && mv $(TESTS)/$(1).running $(TESTS)/$(1).passed) || (mv $(TESTS)/$(1).running $(TESTS)/$(1).failed && echo "           $(COLOR_RED)$(1) FAILED$(COLOR_RESET)" && cat $(TESTS)/$(1).failed && echo "           $(COLOR_RED)$(1) FAILED$(COLOR_RESET)" && false))

TEST_$(1)_DEPS := $$(foreach lib,$(2),$$(if $$(LIB_$$(lib)_DEPS),$$(LIB_$$(lib)_DEPS),$$(error variable LIB_$$(lib)_DEPS for library $(lib) in test $(1) is empty)))

$(TESTS)/$(1).passed:	$(CWD)/$(1).js $$(TEST_$(1)_DEPS) $(NODE_TEST_DEPS)
	$$(if $(verbose_build),@echo '$$(TEST_$(1)_COMMAND)',@echo "[TESTCASE] $(1)")
	@$$(TEST_$(1)_COMMAND)
	$$(if $(verbose_build),@echo '$$(TEST_$(1)_COMMAND)',@echo "           $(COLOR_GREEN)$(1) passed$(COLOR_RESET)")

$(1):	$(CWD)/$(1).js $$(TEST_$(1)_DEPS)
	NODE_PATH=$(NODE_PATH) $(NODE) $(3) $(CWD)/$(1).js

.PHONY: $(1)

$(if $(findstring manual,$(5)),,test $(CURRENT_TEST_TARGETS) $$(CURRENT)_test) $(4):	$(TESTS)/$(1).passed

endef

# vows test case for node

# $(1) name of the test (the javascript file that contains the test case)
# $(2) node.js modules on which it depends
# $(3) options to the vows executable
# $(4) test target
# $(5) test options (eg, manual)

define vowsjs_test
$$(if $(trace),$$(warning called nodejs_test "$(1)" "$(2)" "$(3)"))

TEST_$(1)_COMMAND := rm -f $(TESTS)/$(1).{passed,failed} && ((set -o pipefail && NODE_PATH=$(NODE_PATH) $(NODE) $(3) $(VOWS) $(CWD)/$(1).js > $(TESTS)/$(1).running 2>&1 && mv $(TESTS)/$(1).running $(TESTS)/$(1).passed) || (mv $(TESTS)/$(1).running $(TESTS)/$(1).failed && echo "           $(COLOR_RED)$(1) FAILED$(COLOR_RESET)" && cat $(TESTS)/$(1).failed && echo "           $(COLOR_RED)$(1) FAILED$(COLOR_RESET)" && false))

TEST_$(1)_DEPS := $$(foreach lib,$(2),$$(if $$(LIB_$$(lib)_DEPS),$$(LIB_$$(lib)_DEPS),$$(error variable LIB_$$(lib)_DEPS for library $(lib) in test $(1) is empty)))

$(TESTS)/$(1).passed:	$(CWD)/$(1).js $$(TEST_$(1)_DEPS) $(VOWS_TEST_DEPS)
	$$(if $(verbose_build),@echo '$$(TEST_$(1)_COMMAND)',@echo "[TESTCASE] $(1)")
	@$$(TEST_$(1)_COMMAND)
	$$(if $(verbose_build),@echo '$$(TEST_$(1)_COMMAND)',@echo "           $(COLOR_GREEN)$(1) passed$(COLOR_RESET)")

$(1):	$(CWD)/$(1).js $$(TEST_$(1)_DEPS)
	NODE_PATH=$(NODE_PATH) $(NODE) $(3) $(VOWS) $(CWD)/$(1).js

.PHONY: $(1)

$(if $(findstring manual,$(5)),,test $(CURRENT_TEST_TARGETS) $$(CURRENT)_test) $(4):	$(TESTS)/$(1).passed

endef

endif