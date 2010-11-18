ifeq ($(NODEJS_ENABLED),1)

NODE ?= node
NODE_PRELOAD ?= LD_PRELOAD=$(BIN)/libexception_hook.so
VOWS ?= /usr/local/bin/vows
NODE_PATH := $(if $(NODE_PATH),$(NODE_PATH):)$(BIN)
NODE_TEST_DEPS ?= $(BIN)/libexception_hook.so
VOWS_TEST_DEPS ?= $(NODE_TEST_DEPS)

all compile:	nodejs_programs

# add a node.js addon
# $(1): name of the addon
# $(2): source files to include in the addon
# $(3): libraries to link with
# $(4): other node.js addons that need to be linked in with this one

define nodejs_addon
$$(eval $$(call library,$(1)_node_impl,$(2),node_exception_tracing $(3) $$(foreach lib,$(4),$$(lib)_node_impl),,.so,[NODEJS]))

nodejs_addons: $$(LIB_$(1)_node_impl_DEPS) $$(BIN)/$(1).node

$$(BIN)/$(1).node: $$(LIB_$(1)_node_impl_SO) $$(BIN)/lib$(1)_node_impl.so $$(foreach addon,$(4),$$(BIN)/$$(addon).node) 
	@$$(CXX) $$(CXXFLAGS) $$(CXXLIBRARYFLAGS) -o $$@~ $$(BIN)/lib$(1)_node_impl.so
	@mv $$@~ $$@

endef

# node test case

# $(1) name of the test (the javascript file that contains the test case)
# $(2) node.js modules on which it depends
# $(3) options to the node executable
# $(4) test name
# $(5) test options

define nodejs_test
$$(if $(trace),$$(warning called nodejs_test "$(1)" "$(2)" "$(3)" "$(4)" "$(5)"))

TEST_$(1)_COMMAND := rm -f $(TESTS)/$(1).{passed,failed} && ((set -o pipefail && NODE_PATH=$(NODE_PATH) $(NODE_PRELOAD) $(NODE) $(3) $(CWD)/$(1).js > $(TESTS)/$(1).running 2>&1 && mv $(TESTS)/$(1).running $(TESTS)/$(1).passed) || (mv $(TESTS)/$(1).running $(TESTS)/$(1).failed && echo "           $(COLOR_RED)$(1) FAILED$(COLOR_RESET)" && cat $(TESTS)/$(1).failed && echo "           $(COLOR_RED)$(1) FAILED$(COLOR_RESET)" && false))

TEST_$(1)_DEPS := $$(foreach lib,$(2),$$(BIN)/$$(lib).node)

$(TESTS)/$(1).passed:	$(CWD)/$(1).js $$(TEST_$(1)_DEPS) $(NODE_TEST_DEPS)
	$$(if $(verbose_build),@echo '$$(TEST_$(1)_COMMAND)',@echo "[TESTCASE] $(1)")
	@$$(TEST_$(1)_COMMAND)
	$$(if $(verbose_build),@echo '$$(TEST_$(1)_COMMAND)',@echo "           $(COLOR_GREEN)$(1) passed$(COLOR_RESET)")

$(1):	$(CWD)/$(1).js $$(TEST_$(1)_DEPS)
	NODE_PATH=$(NODE_PATH) $(NODE_PRELOAD) $(NODE) $(3) $(CWD)/$(1).js

.PHONY: $(1)

$(if $(findstring manual,$(5)),,test $(CURRENT_TEST_TARGETS) $$(CURRENT)_test) $(4):	$(TESTS)/$(1).passed

endef


# $(1) name of the program (the javascript file that contains the main file) without the .js
# $(2) node.js modules on which it depends
# $(3) options to the node executable

define nodejs_program

NODE_PROGRAM_$(1)_DEPS := $$(foreach lib,$(2),$$(BIN)/$$(lib).node)

$(BIN)/$(1):	$(CWD)/$(1).js $$(NODE_PROGRAM_$(1)_DEPS) $(NODE_TEST_DEPS)
	@echo "[NODEJS] $(1)"
	@echo "#!/usr/bin/env bash" > $$@~
	@echo -n "//usr/bin/env NODE_PATH=$(NODE_PATH) $(NODE_PRELOAD) $(NODE) " >> $$@~
	@echo -n $$$$ >> $$@~
	@echo -n '0 "' >> $$@~
	@echo -n $$$$ >> $$@~
	@echo '@"; exit' >> $$@~
	@cat $(CWD)/$(1).js >> $$@~
	@chmod +x $$@~
	@mv $$@~ $$@

run_$(1):	$(BIN)/$(1)
	$(BIN)/$(1)

nodejs_programs programs $(1): $(BIN)/$(1)

endef

# vows test case for node

# $(1) name of the test (the javascript file that contains the test case)
# $(2) node.js modules on which it depends
# $(3) options to the vows executable
# $(4) test target
# $(5) test options (eg, manual)

define vowsjs_test
$$(if $(trace),$$(warning called nodejs_test "$(1)" "$(2)" "$(3)"))

TEST_$(1)_COMMAND := rm -f $(TESTS)/$(1).{passed,failed} && ((set -o pipefail && NODE_PATH=$(NODE_PATH) $(NODE_PRELOAD) $(NODE) $(3) $(VOWS) $(CWD)/$(1).js > $(TESTS)/$(1).running 2>&1 && mv $(TESTS)/$(1).running $(TESTS)/$(1).passed) || (mv $(TESTS)/$(1).running $(TESTS)/$(1).failed && echo "           $(COLOR_RED)$(1) FAILED$(COLOR_RESET)" && cat $(TESTS)/$(1).failed && echo "           $(COLOR_RED)$(1) FAILED$(COLOR_RESET)" && false))

TEST_$(1)_DEPS := $$(foreach lib,$(2),$$(BIN)/$$(lib).node)

$(TESTS)/$(1).passed:	$(CWD)/$(1).js $$(TEST_$(1)_DEPS) $(VOWS_TEST_DEPS)
	$$(if $(verbose_build),@echo '$$(TEST_$(1)_COMMAND)',@echo "[TESTCASE] $(1)")
	@$$(TEST_$(1)_COMMAND)
	$$(if $(verbose_build),@echo '$$(TEST_$(1)_COMMAND)',@echo "           $(COLOR_GREEN)$(1) passed$(COLOR_RESET)")

$(1):	$(CWD)/$(1).js $$(TEST_$(1)_DEPS)
	NODE_PATH=$(NODE_PATH) $(NODE_PRELOAD) $(NODE) $(3) $(VOWS) $(CWD)/$(1).js

.PHONY: $(1)

$(if $(findstring manual,$(5)),,test $(CURRENT_TEST_TARGETS) $$(CURRENT)_test) $(4):	$(TESTS)/$(1).passed

endef

endif