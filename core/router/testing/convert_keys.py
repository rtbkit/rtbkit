#!/usr/bin/python
import redis
import time
import argparse
import string
import sys

if len(sys.argv < 3):
       raise Exception("expected source and target hosts as arguments")
r = redis.Redis(host=sys.argv[1], port=6379)
newR = redis.Redis(host=sys.argv[2], port=6379)
print "Connected to Redis...retrieving keys"
campaigns = r.keys('*-budgets')
print "retrieved all keys"
for c in campaigns:
	print(c)
	newCampaign = string.replace(c,"-budgets","")
	strategies = r.hgetall(c)
	campaignKey = "campaigns:" + newCampaign 
	campaignAvailable = 0
	campaignTransferred = 0	
	for key,value in dict.items(strategies):
		#print "\t", key, value 
		strategy = campaignKey + ":" + key
		campaignTransferred += long(value)
		print "strategy:", strategy	
		sdict ={}
		sdict['available'] = 0
		sdict['transferred'] = value
		sdict['spent'] = value
		newR.hmset(strategy, sdict)
	print newCampaign, " transferred: ", campaignTransferred
	cdict = {}
	cdict['available'] = 0
	cdict['transferred'] = campaignTransferred	
	print "command:", cdict
	newR.hmset(campaignKey, cdict)
