#------------------------------------------------------------------------------#
# gc_testing.mk
# Rémi Attab, 01 Feb 2013
# Copyright (c) 2013 Datacratic.  All rights reserved.
#
# Gc's test makefiles
#------------------------------------------------------------------------------#

$(eval $(call test,gc_test,gc,boost))
$(eval $(call test,rcu_protected_test,gc,boost timed))

