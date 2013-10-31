#
# bidder.py
# 
#
import ujson
import pprint
from lwrtb import lwrtb


agent_config = "{\"lossFormat\":\"lightweight\",\"winFormat\":\"full\",\"test\":false,\"minTimeAvailableMs\":5,\"account\":[\"hello\",\"world\"],\"bidProbability\":0.1000000014901161,\"creatives\":[{\"format\":\"728x90\",\"id\":2,\"name\":\"LeaderBoard\"},{\"format\":\"160x600\",\"id\":0,\"name\":\"LeaderBoard\"},{\"format\":\"300x250\",\"id\":1,\"name\":\"BigBox\"}],\"errorFormat\":\"lightweight\",\"externalId\":0}";

proxy_config = "{\"installation\":\"rtb-test\",\"location\":\"mtl\",\"zookeeper-uri\":\"localhost:2181\",\"portRanges\":{\"logs\":[16000,17000],\"router\":[17000,18000],\"augmentors\":[18000,19000],\"configuration\":[19000,20000],\"postAuctionLoop\":[20000,21000],\"postAuctionLoopAgents\":[21000,22000],\"banker.zmq\":[22000,23000],\"banker.http\":9985,\"agentConfiguration.zmq\":[23000,24000],\"agentConfiguration.http\":9986,\"monitor.zmq\":[24000,25000],\"monitor.http\":9987,\"adServer.logger\":[25000,26000]}}";

def error_cb (agent, msg, msgvec):
	print 'ERROR', msg
	pass

def bidreq_cb (agent, ts, bid_id, bid_req, bids, time_left, augmentations, wcm):
	x = ujson.loads(bid_req)
	y= ujson.loads(bids)
	for bid in y['bids']:
		bid['price']='100USD/1M'
		bid['creative']=bid['availableCreatives'][0]
	# print ujson.dumps(y)
	agent.doBid(bid_id, ujson.dumps(y))
	print 'BIDREQ bid on id=%s'%bid_id
	pass

def delivery_cb (agent, dlv):
	pass

def bidresult_cb (agent, res):
	if res.result == lwrtb.WIN: r='WIN'
	elif res.result == lwrtb.LOSS: r='LOSS'
	elif res.result == lwrtb.TOOLATE: r='TOOLATE'
	elif res.result == lwrtb.LOSTBID: r='LOSTBID'
	elif res.result == lwrtb.DROPPEDBID: r='DROPPEDBID'
	elif res.result == lwrtb.NOBUDGET: r='NOBUDGET'
        else: r='BUG'
        print 'RESULT: %s  (auctionId=%s)'%(r,res.auctionId)
	pass

bob = lwrtb.Bidder("BOB", proxy_config)

bob.setBidRequestCb (bidreq_cb)
bob.setErrorCb      (error_cb)
bob.setDeliveryCb   (delivery_cb)
bob.setBidResultCb  (bidresult_cb)
bob.init()
bob.doConfig(agent_config)
bob.start(True)
