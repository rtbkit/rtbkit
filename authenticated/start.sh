#!/usr/bin/env bash

cd ..

TRUE=0
FALSE=1

is_running() {
  if ps ax | grep -v grep | grep $1 > /dev/null
  then
    return $TRUE
  else
    return $FALSE
  fi
}

start_zookeeper() {
  if is_running zookeeper; then
    echo zookeeper already running
  else
    cd /opt/rtbkit/rtbkit/
    /opt/local/bin/zookeeper/bin/zkserver.sh start
  fi
}

start_redis() {
  if is_running redis-server; then
     echo redis-server already running
  else
     echo starting redis-server...
     /opt/local/bin/redis-server /opt/rtbkit/rtbkit/sample.redis.conf &
  fi
}

start_graphite() {
  if is_running graphite; then
    echo graphite is already running
  else
    echo starting graphite...
    cd /opt/graphite/conf/
    for i in `ls *`; do sudo cp $i `echo $i | sed "s/\.example//g"`; done
    /opt/graphite/bin/carbon-cache.py start
  fi
}

start_mock_exchange() {
  if is_running mock_exchange_runner; then
    echo mock exchange is already running
  else
    ./build/x86_64/bin/mock_exchange_runner &
  fi
}

start_rtbkit() {
  ./build/x86_64/bin/launcher --node localhost --script ./launch.sh /opt/authenticated/launch.json
  ./launch.sh
}

start_zookeeper
start_redis
start_graphite
start_mock_exchange
start_rtbkit

## query the accounts and set an initial budget for account "hello"
## curl http://localhost:9985/v1/accounts
## curl http://localhost:9985/v1/accounts/hello/budget -d '{ "USD/1M": 123456789 }'
#
## to re-open the tmux window
## tmux attach
