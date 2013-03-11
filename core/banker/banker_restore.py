#!/usr/bin/python
# Wolfgang Sourdeau - march 2003
# Copyright (c) 2013 Datacratic.  All rights reserved.

# restore script for the Banker database

# note: this scripts does a destructive restore

import redis
import json
import sys

def empty_db(r):
    storedKeys = tuple(r.keys())
    apply(redis.Redis.delete, (r,) + storedKeys)

    storedKeys = r.keys()
    if len(storedKeys) > 0:
        raise Exception("database was not properly emptied")

def load_json(filename):
    inf = open(filename)
    d = json.loads(inf.read())
    inf.close()

    return d

def fill_db(r, d):
    account_names = []
    for key in d:
        value = d[key]
        if type(value) == dict:
            r.hmset(key, value)
        elif isinstance(value, basestring):
            if key.startswith("banker-"):
                account_names.append(key[7:])
            r.set(key, value)
        else:
            raise Exception("unsupported type '%s'" % str(type(key)))

    # TODO: backup files do not contain "banker:accounts" so we need to
    # rebuild it, which is wrong
    if len(account_names) > 0:
        apply(redis.Redis.sadd, (r, "banker:accounts") + tuple(account_names))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("missing filename parameter")

    r = redis.Redis(host='localhost', port=6380)

    empty_db(r)
    d = load_json(sys.argv[1])
    fill_db(r, d)
