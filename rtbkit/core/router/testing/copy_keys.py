#!/usr/bin/python
import redis
import time
import argparse
import string
import sys

if len(sys.argv < 3):
       raise Exception("expected source and target hosts as arguments")

# Copy Keys from one redis database to another
r = redis.Redis(host=sys.argv[1], port=6379)
newR = redis.Redis(host=sys.argv[2], port=6379)
print "Connected to Redis...retrieving keys"
campaigns = r.keys('campaigns:*')
print "retrieved all keys"
for c in campaigns:
       print(c)
       values = r.hgetall(c)
       newR.hmset(c,values)

