ifeq ($(PYTHON_ENABLED),1)

# add a swig wrapper source file
# $(1): filename of source file
# $(2): basename of the filename
define add_swig_source
ifneq ($(PREMAKE),1)
$(if $(trace),$$(warning called add_swig_source "$(1)" "$(2)"))

BUILD_$(OBJ)/$(CWD)/$(2)_wrap.cxx_COMMAND := swig -python -c++  -MMD -MF $(OBJ)/$(CWD)/$(2).d -MT "$(OBJ)/$(CWD)/$(2)_wrap.cxx $(OBJ)/$(CWD)/$(2).lo" -o $(OBJ)/$(CWD)/$(2)_wrap.cxx~ $(SRC)/$(CWD)/$(1)

# Call swig to generate the source file
$(OBJ)/$(CWD)/$(2)_wrap.cxx:	$(SRC)/$(CWD)/$(1)
	@mkdir -p $(OBJ)/$(CWD)
	$$(if $(verbose_build),@echo $$(BUILD_$(OBJ)/$(CWD)/$(2)_wrap.cxx_COMMAND),@echo "[SWIG python] $(CWD)/$(1)")
	@$$(BUILD_$(OBJ)/$(CWD)/$(2)_wrap.cxx_COMMAND)
	@mv $$@~ $$@

# We use the add_c++_source to do most of the work, then simply point
# to the file
$$(eval $$(call add_c++_source,$(2)_wrap.cxx,$(2)_wrap,$(OBJ),-I$(PYTHON_INCLUDE_PATH)))

# Point to the object file produced by the previous macro
BUILD_$(CWD)/$(2).lo_OBJ  := $$(BUILD_$(CWD)/$(2)_wrap.lo_OBJ)

-include $(OBJ)/$(CWD)/$(2).d

endif
endef

# python test case

# $(1) name of the test
# $(2) python modules on which it depends
# $(3) test style.  Currently unused.

define pytest
ifneq ($(PREMAKE),1)
$$(if $(trace),$$(warning called pytest "$(1)" "$(2)" "$(3)"))

TEST_$(1)_COMMAND := rm -f $(TESTS)/$(1).{passed,failed} && ((set -o pipefail && $(PYTHON) $(CWD)/$(1).py > $(TESTS)/$(1).running 2>&1 && mv $(TESTS)/$(1).running $(TESTS)/$(1).passed) || (mv $(TESTS)/$(1).running $(TESTS)/$(1).failed && echo "           $(1) FAILED" && cat $(TESTS)/$(1).failed && false))

$(TESTS)/$(1).passed:	$(CWD)/$(1).py $$(foreach lib,$(2),$$(PYTHON_$$(lib)_DEPS))
	$$(if $(verbose_build),@echo '$$(TEST_$(1)_COMMAND)',@echo "[TESTCASE] $(1)")
	@$$(TEST_$(1)_COMMAND)

$(1):	$(CWD)/$(1).py $$(foreach lib,$(2),$$(PYTHON_$$(lib)_DEPS))
	$(PYTHON) $(CWD)/$(1).py

.PHONY: $(1)

$(if $(findstring manual,$(3)),,test $(CURRENT_TEST_TARGETS) $$(CURRENT)_test) $(4) python_test:	$(TESTS)/$(1).passed
endif
endef

# $(1): name of python file
# $(2): name of directory to go in

define install_python_file
ifneq ($(PREMAKE),1)

$$(if $(trace),$$(warning called install_python_file "$(1)" "$(2)"))

$(BIN)/$(2)/$(1):	$(CWD)/$(1) $(BIN)/$(2)/.dir_exists
	$$(if $(verbose_build),@echo "cp $$< $$@",@echo "[PYTHON INSTALL] $(2)/$(1)")
	cp $$< $$@~
	@mv $$@~ $$@

#$$(w arning building $(BIN)/$(2)/$(1))

all compile: $(BIN)/$(2)/$(1)

endif
endef

# $(1): name of python module
# $(2): list of python source files to copy
# $(3): libraries it depends upon

define python_module
ifneq ($(PREMAKE),1)
$$(if $(trace),$$(warning called python_module "$(1)" "$(2)" "$(3)"))

$$(foreach file,$(2),$$(eval $$(call install_python_file,$$(file),$(1))))

PYTHON_$(1)_DEPS := $$(foreach file,$(2),$(BIN)/$(1)/$$(file)) $$(foreach lib,$(3),$$(LIB_$$(lib)_DEPS))

#$$(w arning PYTHON_$(1)_DEPS=$$(PYTHON_$(1)_DEPS))

python_modules: $$(PYTHON_$(1)_DEPS)

all compile:	python_modules
endif
endef


endif
