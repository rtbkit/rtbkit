$(eval $(call library,analytics,analytics.cc,services))
$(eval $(call program,analytics_runner,analytics boost_program_options))
$(eval $(call program,send_win,analytics))

