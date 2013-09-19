#------------------------------------------------------------------------------#
# nodeangents.mk
# Rémi Attab, 19 Sep 2013
# Copyright (c) 2013 Datacratic.  All rights reserved.
#
# Makefile for the node.js agents examples
#------------------------------------------------------------------------------#

$(eval $(call nodejs_module,budget-controller,budget-controller.js))
$(eval $(call nodejs_module,nodebidagent-config,nodebidagent-config.js))
$(eval $(call nodejs_program,nodebidagent,nodebidagent.js,rtb services budget-controller nodebidagent-config))
