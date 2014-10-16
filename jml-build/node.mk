ifeq ($(NODEJS_ENABLED),1)

NODE ?= $(if $(NODE_DEBUG),node_g,node)
NPM ?= npm
NODE_V8_LIB ?= $(if $(NODE_DEBUG),node-v8_g,node-v8)
NODE_PRELOAD ?= LD_PRELOAD=$(LIB)/libexception_hook.so
VOWS ?= /usr/local/bin/vows
NODE_PATH := $(if $(NODE_PATH),$(NODE_PATH):)$(BIN)
NODE_TEST_DEPS ?= $(LIB)/libexception_hook.so
VOWS_TEST_DEPS ?= $(NODE_TEST_DEPS)

all compile:	nodejs_programs nodejs_addons nodejs_libraries

.PHONY: 	all compile nodejs_addons nodejs_programs nodejs_libraries



# Make sure the target exists even if there are no nodejs_program targets.
nodejs_programs:

# Make sure npm uses the system CAs, not its (outdated) internal one
nodejs_dependencies: package.json
	$(NPM) config set ca ""
	$(NPM) install .

dependencies: nodejs_dependencies

$(BIN)/node_runner: $(BIN)/.dir_exists
	@echo '#!/bin/bash' > $@~
	@echo 'exec /usr/bin/env NODE_PATH=$(NODE_PATH) $(NODE_PRELOAD) $(NODE) "$$@"' >> $@~
	@chmod +x $@~
	@mv $@~ $@

# Dependencies for a single node addon
# $(1): name of the addon
# $(2): target being built (for error message)
node_addon_deps1 = $(if $(findstring x1x,x$(PREMAKE)x),$(NODE_$(1)_DEPS),$(if $(NODE_$(1)_DEPS),$(NODE_$(1)_DEPS),$(error no deps for node library $(1) building $(2) premake $(PREMAKE))))

# Dependencies for a list of node addons
# $(1): list of addons
# $(2): target being built (for error message)
node_addon_deps = $(foreach addon,$(1),$(call node_addon_deps1,$(addon),$(2)))


# add a node.js addon
# $(1): name of the addon
# $(2): source files to include in the addon
# $(3): libraries to link with
# $(4): other node.js addons that need to be linked in with this one

define nodejs_addon
$$(eval $$(call library,$(1)_node_impl,$(2),node_exception_tracing $(3) $$(foreach lib,$(4),$$(lib)_node_impl),,.so,"  $(COLOR_YELLOW)[NODEJS_ADDON]$(COLOR_RESET)"))

NODE_$(1)_DEPS := $$(BIN)/$(1).node $$(call node_addon_deps,$(4))

ifneq ($$(PREMAKE),1)

NODE_$(1)_LINK := $$(BIN)/$(1).node

nodejs_addons: $$(LIB_$(1)_node_impl_DEPS) $$(BIN)/$(1).node

$$(BIN)/$(1).node: $$(LIB_$(1)_node_impl_SO) $$(LIB)/lib$(1)_node_impl.so $$(call node_addon_deps,$(4))
	@$$(CXX) $$(CXXFLAGS) $$(CXXLIBRARYFLAGS) -o $$@~ -l$(1)_node_impl
	@mv $$@~ $$@
endif

endef


# functions for installing JS files from .js or .coffee
install_js_from.js = @cp $(1) $(2)
install_js_from.coffee = @$(COFFEE) --print --compile $(1) > $(2)


# add a node.js module
# $(1): name of the module
# $(2): filename (.js or .coffee)
# $(3): other node.js addons to link with this one

define nodejs_module

NODE_$(1)_DEPS := $(BIN)/$(1).js $$(call node_addon_deps,$(3))

ifneq ($$(PREMAKE),1)

nodejs_libraries $(1): $(BIN)/$(1).js

$(BIN)/$(1).js: $(CWD)/$(2) $$(call node_addon_deps,$(3)) $(BIN)/.dir_exists
	@echo " $(COLOR_YELLOW)[NODEJS_MODULE]$(COLOR_RESET) $(1)"
	$$(if $$(install_js_from$(suffix $(2))),,$$(error js suffix $(suffix $(2)) unknown))
	$$(call install_js_from$(suffix $(2)), $$<, $$@~)
	@mv $$@~ $$@

endif

endef

# node test case

# $(1) name of the test (the javascript file that contains the test case)
# $(2) node.js modules on which it depends
# $(3) options to the node executable
# $(4) test name
# $(5) test options

define nodejs_test
$$(if $(trace),$$(warning called nodejs_test "$(1)" "$(2)" "$(3)" "$(4)" "$(5)"))

TEST_$(1)_DEPS := $$(call node_addon_deps,$(2),$(1))

ifneq ($$(PREMAKE),1)

TEST_$(1)_COMMAND := rm -f $(TESTS)/$(1).{passed,failed} && ((set -o pipefail && NODE_PATH=$(NODE_PATH) $(NODE_PRELOAD) $(NODE) $(3) $(CWD)/$(1).js > $(TESTS)/$(1).running 2>&1 && mv $(TESTS)/$(1).running $(TESTS)/$(1).passed) || (mv $(TESTS)/$(1).running $(TESTS)/$(1).failed && echo "                 $(COLOR_RED)$(1) FAILED$(COLOR_RESET)" && cat $(TESTS)/$(1).failed && echo "           $(COLOR_RED)$(1) FAILED$(COLOR_RESET)" && false))

$(TESTS)/$(1).passed:	$(TESTS)/.dir_exists $(CWD)/$(1).js $$(TEST_$(1)_DEPS) $(NODE_TEST_DEPS)
	$$(if $(verbose_build),@echo '$$(TEST_$(1)_COMMAND)',@echo "      $(COLOR_VIOLET)[TESTCASE]$(COLOR_RESET) $(1)")
	@$$(TEST_$(1)_COMMAND)
	$$(if $(verbose_build),@echo '$$(TEST_$(1)_COMMAND)',@echo "                 $(COLOR_GREEN)$(1) passed$(COLOR_RESET)")

$(1):	$(CWD)/$(1).js $$(TEST_$(1)_DEPS) $(NODE_TEST_DEPS)
	NODE_PATH=$(NODE_PATH) $(NODE_PRELOAD) $(NODE) $(3) $(CWD)/$(1).js

.PHONY: $(1)

$(if $(findstring manual,$(5)),manual,test $(if $(findstring noauto,$(5)),,autotest) ) $(CURRENT_TEST_TARGETS) $$(CURRENT)_test $(4):	$(TESTS)/$(1).passed
endif

endef

# functions for installing JS files from .js or .coffee
append_js_from.js = @cat $(1) >> $(2)
append_js_from.coffee = @$(COFFEE) --print --compile $(1) >> $(2)

# $(1) name of the program
# $(2) filename (.js or .coffee)
# $(3) node.js modules on which it depends

define nodejs_program

ifneq ($$(PREMAKE),1)
$(BIN)/$(1):	$(BIN)/node_runner $(CWD)/$(2) $$(call node_addon_deps,$(3),$(1)) $(NODE_TEST_DEPS)
	@echo "$(COLOR_BLUE)[NODEJS_PROGRAM]$(COLOR_RESET) $(1)"
	@echo "#!/bin/bash $(BIN)/node_runner" > $$@~
	$$(if $$(append_js_from$(suffix $(2))),,$$(error js suffix $(suffix $(2)) unknown))
	$$(call append_js_from$(suffix $(2)), $(CWD)/$(2), $$@~)
	@chmod +x $$@~
	@mv $$@~ $$@

run_$(1):	$(BIN)/$(1)
	$(BIN)/$(1)  $($(1)_ARGS)

nodejs_programs programs $(1): $(BIN)/$(1)
endif

endef

# vows test case for node

# $(1) name of the test (the javascript file that contains the test case)
# $(2) node.js modules on which it depends
# $(3) options to the vows executable
# $(4) test target
# $(5) test options (eg, manual)

define vowsjs_test
$$(if $(trace),$$(warning called nodejs_test "$(1)" "$(2)" "$(3)"))

TEST_$(1)_DEPS := $$(call node_addon_deps,$(2),$(1))

ifneq ($$(PREMAKE),1)
TEST_$(1)_COMMAND := rm -f $(TESTS)/$(1).{passed,failed} && ((set -o pipefail && NODE_PATH=$(NODE_PATH) $(NODE_PRELOAD) $(NODE) $(3) $(VOWS) $(CWD)/$(1).js > $(TESTS)/$(1).running 2>&1 && mv $(TESTS)/$(1).running $(TESTS)/$(1).passed) || (mv $(TESTS)/$(1).running $(TESTS)/$(1).failed && echo "                 $(COLOR_RED)$(1) FAILED$(COLOR_RESET)" && cat $(TESTS)/$(1).failed && echo "           $(COLOR_RED)$(1) FAILED$(COLOR_RESET)" && false))


#$$(w arning TEST_$(1)_DEPS := $$(TEST_$(1)_DEPS))

$(TESTS)/$(1).passed:	$(CWD)/$(1).js $$(TEST_$(1)_DEPS) $(VOWS_TEST_DEPS)
	$$(if $(verbose_build),@echo '$$(TEST_$(1)_COMMAND)',@echo "      $(COLOR_VIOLET)[TESTCASE]$(COLOR_RESET) $(1)")
	@$$(TEST_$(1)_COMMAND)
	$$(if $(verbose_build),@echo '$$(TEST_$(1)_COMMAND)',@echo "                 $(COLOR_GREEN)$(1) passed$(COLOR_RESET)")

$(1):	$(CWD)/$(1).js $$(TEST_$(1)_DEPS)
	NODE_PATH=$(NODE_PATH) $(NODE_PRELOAD) $(NODE) $(3) $(VOWS) $(CWD)/$(1).js

.PHONY: $(1)

$(if $(findstring manual,$(5)),manual,test $(if $(findstring noauto,$(5)),,autotest) ) $(CURRENT_TEST_TARGETS) $$(CURRENT)_test $(4):	$(TESTS)/$(1).passed
endif

endef

# coffeescript test case

# $(1) name of the test (the coffeescript file that contains the test case)
# $(2) node.js modules on which it depends
# $(3) options to the node executable
# $(4) test name
# $(5) test options

define coffee_test
$$(if $(trace),$$(warning called nodejs_test "$(1)" "$(2)" "$(3)" "$(4)" "$(5)"))

TEST_$(1)_DEPS := $$(call node_addon_deps,$(2),$(1))

ifneq ($$(PREMAKE),1)

TEST_$(1)_COMMAND := rm -f $(TESTS)/$(1).{passed,failed} && ((set -o pipefail && NODE_PATH=$(NODE_PATH) $(NODE_PRELOAD) $(NODE) $(3) $(TESTS)/$(1).js > $(TESTS)/$(1).running 2>&1 && mv $(TESTS)/$(1).running $(TESTS)/$(1).passed) || (mv $(TESTS)/$(1).running $(TESTS)/$(1).failed && echo "                 $(COLOR_RED)$(1) FAILED$(COLOR_RESET)" && cat $(TESTS)/$(1).failed && echo "           $(COLOR_RED)$(1) FAILED$(COLOR_RESET)" && false))

$(TESTS)/$(1).passed:	$(TESTS)/.dir_exists $(TESTS)/$(1).js $$(TEST_$(1)_DEPS) $(NODE_TEST_DEPS)
	$$(if $(verbose_build),@echo '$$(TEST_$(1)_COMMAND)',@echo "      $(COLOR_VIOLET)[TESTCASE]$(COLOR_RESET) $(1)")
	@$$(TEST_$(1)_COMMAND)
	$$(if $(verbose_build),@echo '$$(TEST_$(1)_COMMAND)',@echo "                 $(COLOR_GREEN)$(1) passed$(COLOR_RESET)")

$(1):	$(TESTS)/$(1).js $$(TEST_$(1)_DEPS)
	NODE_PATH=$(NODE_PATH) $(NODE_PRELOAD) $(NODE) $(3) $(TESTS)/$(1).js

$(TESTS)/$(1).js: $(TESTS)/.dir_exists $(CWD)/$(1).coffee
	$$(call install_js_from.coffee, $(CWD)/$(1).coffee, $(TESTS)/$(1).js)

.PHONY: $(1)

$(if $(findstring manual,$(5)),manual,test $(if $(findstring noauto,$(5)),,autotest) ) $(CURRENT_TEST_TARGETS) $$(CURRENT)_test $(4):	$(TESTS)/$(1).passed
endif

endef

# vows test case for node written in coffeescript

# $(1) name of the test (the javascript file that contains the test case)
# $(2) node.js modules on which it depends
# $(3) options to the vows executable
# $(4) test target
# $(5) test options (eg, manual)

define vowscoffee_test
$$(if $(trace),$$(warning called nodejs_test "$(1)" "$(2)" "$(3)"))

TEST_$(1)_DEPS := $$(call node_addon_deps,$(2),$(1))

ifneq ($$(PREMAKE),1)
TEST_$(1)_COMMAND := rm -f $(TESTS)/$(1).{passed,failed} && ((set -o pipefail && NODE_PATH=$(NODE_PATH) $(NODE_PRELOAD) $(NODE) $(3) $(VOWS) $(TESTS)/$(1).js > $(TESTS)/$(1).running 2>&1 && mv $(TESTS)/$(1).running $(TESTS)/$(1).passed) || (mv $(TESTS)/$(1).running $(TESTS)/$(1).failed && echo "                 $(COLOR_RED)$(1) FAILED$(COLOR_RESET)" && cat $(TESTS)/$(1).failed && echo "           $(COLOR_RED)$(1) FAILED$(COLOR_RESET)" && false))


#$$(w arning TEST_$(1)_DEPS := $$(TEST_$(1)_DEPS))

$(TESTS)/$(1).passed:	$(TESTS)/$(1).js $$(TEST_$(1)_DEPS) $(VOWS_TEST_DEPS) $(TESTS)/.dir_exists
	$$(if $(verbose_build),@echo '$$(TEST_$(1)_COMMAND)',@echo "      $(COLOR_VIOLET)[TESTCASE]$(COLOR_RESET) $(1)")
	@$$(TEST_$(1)_COMMAND)
	$$(if $(verbose_build),@echo '$$(TEST_$(1)_COMMAND)',@echo "                 $(COLOR_GREEN)$(1) passed$(COLOR_RESET)")

$(1):	$(TESTS)/$(1).js $$(TEST_$(1)_DEPS) $(VOWS_TEST_DEPS)
	NODE_PATH=$(NODE_PATH) $(NODE_PRELOAD) $(NODE) $(3) $(VOWS) $(TESTS)/$(1).js $($(1)_ARGS)

$(TESTS)/$(1).js: $(TESTS)/.dir_exists $(CWD)/$(1).coffee
	$$(call install_js_from.coffee, $(CWD)/$(1).coffee, $(TESTS)/$(1).js)

.PHONY: $(1)

$(if $(findstring manual,$(5)),manual,test $(if $(findstring noauto,$(5)),,autotest) ) $(CURRENT_TEST_TARGETS) $$(CURRENT)_test $(4):	$(TESTS)/$(1).passed
endif

endef

endif
