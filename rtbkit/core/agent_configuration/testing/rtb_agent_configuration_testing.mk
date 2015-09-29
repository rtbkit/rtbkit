# RTB Agent Configuration testing makefile
# JS Bejeau, 25 June 2015

$(eval $(call test,rtb_agent_config_validator_test,agent_configuration,boost))
$(eval $(call test,rtb_fees_test,agent_configuration,boost))


