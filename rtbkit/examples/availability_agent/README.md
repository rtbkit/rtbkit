# Availability Agent #

The availability agent is a agent that never bids on any inventory. Instead it listens on a sample of the bid request traffic and builds a pool of bid request. The user can then query the agent through a REST interface to determine what aproximate QPS you can expect for a given agent configuration. This type of agent is useful to determine whether an agent configuration is too strict for the given traffic.

Note that this is an example which hasn't been fully tested or documented so use at your own risk and peril.
