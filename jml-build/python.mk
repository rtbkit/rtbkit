ifeq ($(PYTHON_ENABLED),1)

PYTHON_VERSION_DETECTED := $(shell $(JML_BUILD)/detect_python.sh)
PYTHON_VERSION ?= $(PYTHON_VERSION_DETECTED)

PYTHON_INCLUDE_PATH ?= $(VIRTUALENV)/include/python$(PYTHON_VERSION)
PYTHON ?= python$(PYTHON_VERSION)
PIP ?= pip
PYFLAKES ?= true

PYTHON_PURE_LIB_PATH ?= $(BIN)
PYTHON_PLAT_LIB_PATH ?= $(BIN)
PYTHON_BIN_PATH ?= $(BIN)

RUN_PYTHONPATH := $(if $(PYTHONPATH),$(PYTHONPATH):,)$(PYTHON_PURE_LIB_PATH):$(PYTHON_PLAT_LIB_PATH):$(PYTHON_BIN_PATH)

PYTHONPATH ?= RUN_PYTHONPATH

ifdef VIRTUALENV

$(VIRTUALENV)/bin/activate:
	virtualenv $(VIRTUALENV)

python_dependencies: $(VIRTUALENV)/bin/activate

PYTHON_EXECUTABLE ?= $(VIRTUALENV)/bin/python

endif

python_dependencies:
	@if [ -f python_requirements.txt ]; then \
		$(PIP) install -r python_requirements.txt; \
	fi

# Loop over the python_extra_requirements.txt file and install packages in
# order. We did that because the package "statsmodels" does not handle
# dependencies that are not installed. For more information, see:
# https://github.com/statsmodels/statsmodels/pull/1902
# https://github.com/statsmodels/statsmodels/issues/1897
#
# WARNING: packages order in python_extra_requirements.txt matters!!!
python_extra_dependencies: python_extra_requirements.txt python_dependencies	
	jml-build/get_python_requirements.sh -c $(PIP) -r python_extra_requirements.txt 

dependencies: python_dependencies

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
# $(3) test options (e.g. manual)
# $(4) test targets

define python_test
ifneq ($(PREMAKE),1)
$$(if $(trace),$$(warning called python_test "$(1)" "$(2)" "$(3)" "$(4)"))

TEST_$(1)_COMMAND := rm -f $(TESTS)/$(1).{passed,failed} && $(PYFLAKES) $(CWD)/$(1).py && ((set -o pipefail && PYTHONPATH=$(RUN_PYTHONPATH) $(PYTHON) $(CWD)/$(1).py > $(TESTS)/$(1).running 2>&1 && mv $(TESTS)/$(1).running $(TESTS)/$(1).passed) || (mv $(TESTS)/$(1).running $(TESTS)/$(1).failed && echo "                 $(COLOR_RED)$(1) FAILED$(COLOR_RESET)" && cat $(TESTS)/$(1).failed && false))

$(TESTS)/$(1).passed:	$(TESTS)/.dir_exists $(CWD)/$(1).py $$(foreach lib,$(2),$$(PYTHON_$$(lib)_DEPS)) $$(foreach pymod,$(2),$(TMPBIN)/$$(pymod)_pymod)
	$$(if $(verbose_build),@echo '$$(TEST_$(1)_COMMAND)',@echo "      $(COLOR_VIOLET)[TESTCASE]$(COLOR_RESET) $(1)")
	@$$(TEST_$(1)_COMMAND)
	$$(if $(verbose_build),@echo '$$(TEST_$(1)_COMMAND)',@echo "                 $(COLOR_GREEN)$(1) passed$(COLOR_RESET)")

$(1):	$(CWD)/$(1).py $$(foreach lib,$(2),$$(PYTHON_$$(lib)_DEPS)) $$(foreach pymod,$(2),$(TMPBIN)/$$(pymod)_pymod)
	@$(PYFLAKES) $(CWD)/$(1).py
	PYTHONPATH=$(RUN_PYTHONPATH) $(PYTHON) $(CWD)/$(1).py $($(1)_ARGS)

.PHONY: $(1)

$(if $(findstring manual,$(3)),manual,test $(if $(findstring noauto,$(3)),,autotest) ) $(CURRENT_TEST_TARGETS) $$(CURRENT)_test $(4) python_test:	$(TESTS)/$(1).passed
endif
endef

# $(1): name of python file
# $(2): name of directory to go in

define install_python_file
ifneq ($(PREMAKE),1)

$$(if $(trace),$$(warning called install_python_file "$(1)" "$(2)"))

$(PYTHON_PURE_LIB_PATH)/$(2)/$(1):	$(CWD)/$(1) $(PYTHON_PURE_LIB_PATH)/$(2)/.dir_exists
	$$(if $(verbose_build),@echo "cp $$< $$@",@echo " $(COLOR_YELLOW)[PYTHON_MODULE]$(COLOR_RESET) $(2)/$(1)")
	@$(PYFLAKES) $$<
	@cp $$< $$@~
	@mv $$@~ $$@

#$$(w arning building $(BIN)/$(2)/$(1))

all compile: $(PYTHON_PURE_LIB_PATH)/$(2)/$(1)

endif
endef

# $(1): name of python module
# $(2): list of python source files to copy
# $(3): python modules it depends upon
# $(4): libraries it depends upon

define python_module
ifneq ($(PREMAKE),1)
$$(if $(trace),$$(warning called python_module "$(1)" "$(2)" "$(3)" "$(4)"))

$$(foreach file,$(2),$$(eval $$(call install_python_file,$$(file),$(1))))

PYTHON_$(1)_DEPS := $$(foreach file,$(2),$(PYTHON_PURE_LIB_PATH)/$(1)/$$(file)) $$(foreach pymod,$(3),$(TMPBIN)/$$(pymod)_pymod) $$(foreach pymod,$(3),$$(PYTHON_$$(pymod)_DEPS)) $$(foreach lib,$(4),$$(LIB_$$(lib)_DEPS))

#$$(w arning PYTHON_$(1)_DEPS=$$(PYTHON_$(1)_DEPS))

$(TMPBIN)/$(1)_pymod: $$(PYTHON_$(1)_DEPS)
	@mkdir -p $$(dir $$@)
	@touch $(TMPBIN)/$(1)_pymod

python_modules: $$(PYTHON_$(1)_DEPS) $(TMPBIN)/$(1)_pymod

all compile:	python_modules
endif
endef

# $(1): name of python program
# $(2): python source file to copy
# $(3): python modules it depends upon

define python_program
ifneq ($(PREMAKE),1)
$$(if $(trace),$$(warning called python_program "$(1)" "$(2)" "$(3)"))

PYTHON_$(1)_DEPS := $(PYTHON_BIN_PATH)/$(1) $$(foreach pymod,$(3),$$(PYTHON_$$(pymod)_DEPS))

.PHONY: run_$(1)

run_$(1):	$(PYTHON_BIN_PATH)/$(1)
	$(PYTHON) $(PYTHON_BIN_PATH)/$(1)  $($(1)_ARGS)

$(PYTHON_BIN_PATH)/$(1): $(CWD)/$(2) $(PYTHON_BIN_PATH)/.dir_exists $$(foreach pymod,$(3),$(TMPBIN)/$$(pymod)_pymod) $$(foreach pymod,$(3),$$(PYTHON_$$(pymod)_DEPS))
	@echo "$(COLOR_BLUE)[PYTHON_PROGRAM]$(COLOR_RESET) $(1)"
	@$(PYFLAKES) $$<
	@(echo "#!$(PYTHON_EXECUTABLE)"; cat $$<) > $$@~
	@chmod +x $$@~
	@mv $$@~ $$@

#$$(w arning PYTHON_$(1)_DEPS=$$(PYTHON_$(1)_DEPS))

$(1): $(PYTHON_BIN_PATH)/$(1)

python_programs: $$(PYTHON_$(1)_DEPS)

all compile:	python_programs
endif
endef

# add a python addon
# $(1): name of the addon
# $(2): source files to include in the addon
# $(3): libraries to link with

define python_addon
$$(eval $$(call set_compile_option,$(2),-I$$(PYTHON_INCLUDE_PATH)))
$$(eval $$(call library,$(1),$(2),$(3) boost_python,$(1),,"  $(COLOR_YELLOW)[PYTHON_ADDON]$(COLOR_RESET)"))

ifneq ($(PREMAKE),1)

ifneq ($(LIB),$(PYTHON_PLAT_LIB_PATH))
$(PYTHON_PLAT_LIB_PATH)/$(1).so:	$(LIB)/$(1).so
	@cp $$< $$@~ && mv $$@~ $$@
endif


$(TMPBIN)/$(1)_pymod: $(PYTHON_PLAT_LIB_PATH)/$(1).so
	@mkdir -p $$(dir $$@)
	@touch $(TMPBIN)/$(1)_pymod

python_modules: $(PYTHON_PLAT_LIB_PATH)/$(1).so

endif
endef

# adds a C++ program as a dependency of a python test script
# $(1): name of the python test script
# $(2): list of C++ dependencies

define add_python_test_dep
ifneq ($(PREMAKE),1)

$(CWD)/$(1).py: $$(foreach dep,$(2),$(BIN)/$$(dep))

endif
endef

endif # ifeq ($(PYTHON_ENABLED),1)
