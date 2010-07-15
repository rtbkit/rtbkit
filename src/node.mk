ifeq ($(NODEJS_ENABLED),1)

NODE ?= node
NODE_PATH := $(if $(NODE_PATH),$(NODE_PATH):)$(BIN)

# add a node.js addon
# $(1): name of the addon
# $(2): source files to include in the addon
# $(3): libraries to link with

define nodejs_addon
$$(eval $$(call library,$(1),$(2),$(3),$(1),.node,[NODEJS]))
endef

# node test case

# $(1) name of the test (the javascript file that contains the test case)
# $(2) node.js modules on which it depends
# $(3) test style.  Currently unused.

define nodejs_test
$$(if $(trace),$$(warning called nodejs_test "$(1)" "$(2)" "$(3)"))

TEST_$(1)_COMMAND := rm -f $(TESTS)/$(1).{passed,failed} && ((set -o pipefail && NODE_PATH=$(NODE_PATH) $(NODE) $(CWD)/$(1).js > $(TESTS)/$(1).running 2>&1 && mv $(TESTS)/$(1).running $(TESTS)/$(1).passed) || (mv $(TESTS)/$(1).running $(TESTS)/$(1).failed && echo "           $(1) FAILED" && cat $(TESTS)/$(1).failed && false))

$(TESTS)/$(1).passed:	$(CWD)/$(1).js $$(foreach lib,$(2),$$(LIB_$$(lib)_DEPS))
	$$(if $(verbose_build),@echo '$$(TEST_$(1)_COMMAND)',@echo "[TESTCASE] $(1)")
	@$$(TEST_$(1)_COMMAND)

$(1):	$(CWD)/$(1).js $$(foreach lib,$(2),$$(PYTHON_$$(lib)_DEPS))
	NODE_PATH=$(NODE_PATH) $(NODE) $(CWD)/$(1).js

.PHONY: $(1)

test $(CURRENT_TEST_TARGETS) $(4):	$(TESTS)/$(1).passed

endef

endif