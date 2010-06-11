# stats_testing.mk
# Jeremy Barnes, 9 November 2009
# Copyright (c) 2009 Jeremy Barnes.  All rights reserved.
#
# Testing for stats functionality.

$(eval $(call test,auc_test,stats arch,boost))
$(eval $(call test,rmse_test,stats arch,boost))

