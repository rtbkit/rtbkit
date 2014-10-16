#!/usr/bin/python
# Wolfgang Sourdeau - march 2003
# Copyright (c) 2013 Datacratic.  All rights reserved.

# backup script for the Banker database

# FIXME: this script does not perform the backup atomically, which can lead to
# small inconsistencies between the states of the accounts

import redis
import json
import datetime



r = redis.Redis(host='localhost')

d = {}
for key in r.keys():
    val_type = r.type(key)
    if val_type == "hash":
        d[key] = r.hgetall(key)
    elif val_type == "string":
        d[key] = r.get(key)
    else:
        raise Exception("unhandled value type: %s" % val_type)


filename = "banker_backup_%s.json" % datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
writer = open(filename, "w")
writer.write(json.dumps(d))
writer.close()
