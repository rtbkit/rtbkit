# ----- Copyright (c) 2014 Datacratic. All rights reserved.
"""
This module extends tornados handler classes implementing GET and POST
requests to respond to BID requests

A basic openRtb class helps to interpret requests and prepare responses

Tornado request handlers are extended to handle openRtb, and a
FixedPriceBidder MixIn is used to calculate the bids.
Replacing the Mixin by a smarter strategy using the same HTTP handlers
will create a new bid agent.

There are 2 tornado apps listening at win and event ports
playing the role of a dummy ad server. No action is taken on the events though!

To improve response time a tornado http server is being used to spawn extra
proceses to deal with larger volume of requests.

This is a simplistic implementation and should not be expected to
perform under high load, as it was tested under a few kqps

Currently the average response time for bids is around 14 to 21 ms.


"""
# NOTE: This example is not integrated as part of the rtbkit build system
# as it is an example of how to build a bidder that uses only HTTP interface
# Therefore, TornadoWeb must be manually installed to have this script working
# To do so, execute the following command from your shell:
#    $> pip install tornado
# this will install tornado system wide
# to have tornado installed only at the user level do:
#    $> pip install --user tornado




__version__ = "0.1"
__all__ = ["OpenRTBResponse",
           "FixedPriceBidderMixIn",
           "TornadoDummyRequestHandler",
           "TornadoBaseBidAgentRequestHandler",
           "TornadoFixPriceBidAgentRequestHandler",
           "BudgetControl"]


# IMPORTS

# util libs
from copy import deepcopy
import json
import random

# tornado web
from tornado import process
from tornado import netutil
from tornado import httpserver
from tornado.web import RequestHandler, Application, url
from tornado.ioloop import IOLoop
from tornado.httpclient import AsyncHTTPClient
from tornado.ioloop import PeriodicCallback


# IMPLEMENTATION

# helper function reads the global config obj from file
def read_config(configFile):
    """read config file into config object"""
    cfg = open(configFile)
    contents = json.load(cfg)
    return contents


# this global is used by the bidder class to configure itself
# because the request handler class makes it hard to pass this
# as a argument
CONFIGURATION_FILE = "../http_config.json"
CONFIG_OBJ = read_config(CONFIGURATION_FILE)


# ----- minimalistic OpenRTB response message class

class OpenRTBResponse():
    """this is a helper class to build basic OpenRTB json objects"""

    # field names - constants to avoid magic strings inside the function
    key_id = "id"
    key_bid = "bid"
    key_crid = "crid"
    key_ext = "ext"
    key_extid = "external-id"
    key_ext_creatives = "creative-indexes"
    key_priority = "priority"
    key_impid = "impid"
    key_price = "price"
    key_seatbid = "seatbid"

    # template obejcts
    bid_object = {key_id: "1",
                  key_impid: "1",
                  key_price: 1.0,
                  key_crid: "",
                  key_ext: {key_priority: 1.0}}
    seat_bid_object = {key_bid: [deepcopy(bid_object)]}
    bid_response_object = {key_id: "1",
                           key_seatbid: [deepcopy(seat_bid_object)]}

    def get_empty_response(self):
        """returns an object with the scafold of an rtb response
        but containing only default values"""
        empty_resp = deepcopy(self.bid_response_object)

        return empty_resp

    def get_default_response(self, req):
        """returns an object with the scafold of an rtb response
        and fills some fields based on the request provided"""

        default_resp = None

        if (self.validate_req(req)):
            # since this is a valid request we can return a response
            default_resp = deepcopy(self.bid_response_object)

            # copy request id
            default_resp[self.key_id] = req[self.key_id]

            # empty the bid list (we assume only one seat bid for simplicity)
            default_resp[self.key_seatbid][0][self.key_bid] = []

            # default values for some of the fields of the bid response
            id_counter = 0
            new_bid = deepcopy(self.bid_object)

            # iterate over impressions array from request and
            # populate bid list
            # NOTE: as an example we are bidding on all the impressions,
            # usually that is not what one real bid would look like!!!
            for imp in req["imp"]:
                # -> imp is the field name @ the req

                # dumb bid id, just a counter
                id_counter = id_counter + 1
                new_bid[self.key_id] = str(id_counter)

                # copy impression id as imp for this bid
                new_bid[self.key_impid] = imp[self.key_id]

                externalId = 0
                # try to copy external id to the response
                try:
                    externalId = imp[self.key_ext]["external-ids"][0]
                    new_bid[self.key_ext][self.key_extid] = externalId
                except:
                    externalId = -1  # and do not add this fiel

                # will keep the defaul price as it'll be changed by bidder
                # and append this bid into the bid response
                ref2bidList = default_resp[self.key_seatbid][0][self.key_bid]
                ref2bidList.append(deepcopy(new_bid))

        return default_resp

    def validate_req(self, req):
        """ validates the fields in the request"""
        # not implemented yet. should check if the structure from
        # the request is according to the spec
        # this is just a dummy implementation and we assume everything is fine
        valid = True
        return valid


# ----- simplistic fixed price bidder MixIn class,
#       has to be mixed into a request handler

class FixedPriceBidderMixIn():
    """Dumb bid agent Mixin that bid 100% at $1"""

    # mixins do not have their __init__ (constructor) called
    # so this class do not have it and the load of configuration
    # have to be dealt with by the class that incorporates it!!!

    bid_config = None
    openRtb = OpenRTBResponse()

    def do_config(self, cfgObj):
        self.bid_config = {}
        self.bid_config["probability"] = cfgObj["bidProbability"]
        self.bid_config["price"] = 1.0
        self.bid_config["creatives"] = cfgObj["creatives"]

    def do_bid(self, bid_req):
        # -------------------
        # bid logic:
        # since this is a fix price bidder,
        # just mapping the request to the response
        # and using the default price ($1) will do the work.
        # -------------------

        # assemble defaul response
        resp = self.openRtb.get_default_response(bid_req)

        # ---
        # update bid with price and creatives
        # ---

        # first we need to buid a dictionary
        # that correlates impressions from the request
        # to creative lists
        # FORMAT: dict[extId][impId] = [creat1..creatN]
        impDict = {}
        impList = bid_req["imp"]
        for imp in impList:
            # list of external ids from this impression
            extIdsList = imp[OpenRTBResponse.key_ext]["external-ids"]
            for extId in extIdsList:
                tempDict = {}
                creatives = imp[OpenRTBResponse.key_ext][OpenRTBResponse.key_ext_creatives]
                impId = imp[OpenRTBResponse.key_id]

                tempDict[impId] = creatives[str(extId)]
                impDict[extId] = deepcopy(tempDict)

        # then we iterate over all bids and choose a a random creative for each bid
        # NOTE: we are just doing this fot the first seatbid for simplicity's sake
        ref2seatbid0 = resp[OpenRTBResponse.key_seatbid][0]
        for bid in ref2seatbid0[OpenRTBResponse.key_bid]:

            # update bid price
            bid[OpenRTBResponse.key_price] = self.bid_config["price"]

            # gets the list of creatives from the ext field in the request
            extId = bid[OpenRTBResponse.key_ext][OpenRTBResponse.key_extid]
            impId = bid[OpenRTBResponse.key_impid]
            creativeList = impDict[extId][impId]

            # gets one of the creative indexes randomly
            creatNdx = random.choice(creativeList)

            # get creative id
            creativeId = str(self.bid_config["creatives"][creatNdx]["id"])

            # set the cretive id to the bid
            bid[OpenRTBResponse.key_crid] = creativeId

        return resp


# ----- this dummy handler always answers HTTP 200 to adserver events
#       no further action is taken on the events received

class TornadoDummyRequestHandler(RequestHandler):
    """dummy handler just answer 200. Used to run a dummy adserver"""
    def post(self):
        self.set_status(200)
        self.write("")

    def get(self):
        self.set_status(200)
        self.write("")


# ----- tornado request handler class extend
#       this class is a general bid Agent hadler.
#       bid processing must be implemented in a derived class

class TornadoBaseBidAgentRequestHandler(RequestHandler):
    """ extends tornado handler to answer openRtb requests"""
    def post(self):
        result_body = self.process_req()
        self.write(result_body)

    def get(self):
        result_body = self.process_req()
        self.write(result_body)

    def process_req(self):
        """processes post requests"""

        ret_val = ""

        if self.request.headers["Content-Type"].startswith("application/json"):
            req = json.loads(self.request.body)
        else:
            req = None

        if (req is not None):
            resp = self.process_bid(req)

            if (resp is not None):
                self.set_status(200)
                self.set_header("Content-type", "application/json")
                self.set_header("x-openrtb-version", "2.1")
                ret_val = json.dumps(resp)

            else:
                # print("process_bid error")
                self.set_status(204)
                ret_val = "Error\n"
        else:
            # print("request not json")
            self.set_status(204)
            ret_val = "Error\n"

        # print DEBUG
        # print("req: " + self.request.body)
        # print("resp: " + ret_val)

        return ret_val

    def process_bid(self, req):
        """---TBD in subclass---"""
        resp = None
        return resp


# ----- minimal fixed price bid agent implementation.
#       just extends base request handler class and mix in fix price strategy

class TornadoFixPriceBidAgentRequestHandler(TornadoBaseBidAgentRequestHandler,
                                            FixedPriceBidderMixIn):
    """ This class extends TornadoBaseBidAgentRequestHandler
    The bidding logic is provided by a external object passed as
    parameter to the constructor"""

    def __init__(self, application, request, **kwargs):
        """constructor just call parent INIT and run MixIn's do_config"""
        super(TornadoBaseBidAgentRequestHandler, self).__init__(application, request, **kwargs)
        if (self.bid_config is None):
            # due to the way this class is instantiated
            # we have to use this global var
            self.do_config(CONFIG_OBJ["ACS"]["Config"])

    def process_bid(self, req):
        """process bid request by calling bidder mixin do_bid() method"""
        resp = self.do_bid(req)
        return resp


# ----- callback funtion used by Budget pacer

def handle_async_request(response):
    """ this callback function will handle the response from
    the AsyncHTTPClient call to the banker"""
    if response.error:
        print ("Request Error!")
    else:
        print ("Request response OK")
        print response.body


# ----- Budget allocation class do top up budget for bid agent account

class BudgetControl(object):
    """send rest requests to the banker to pace the budget)"""

    def start(self, cfgObj):
        """config pacer"""
        self.body = '{"USD/1M": ' + str(cfgObj["Banker"]["Budget"]) + '}'
        self.headers = {"Accept": "application/json"}
        self.url = "http://" + cfgObj["Banker"]["Ip"]
        self.url = self.url + ":" + str(cfgObj["Banker"]["Port"])
        acc = cfgObj["ACS"]["Config"]["account"]
        self.url = self.url + "/v1/accounts/" + acc[0]+":"+acc[1] + "/balance"
        self.http_client = AsyncHTTPClient()

        # register with ACS
        self.acs_register(cfgObj["ACS"])

        # call the first budget pace request
        self.http_request()

    def http_request(self):
        """called periodically to updated the budget"""
        try:
            print("Budgeting: " + self.body)
            self.http_client.fetch(self.url, callback=handle_async_request, method='POST', headers=self.headers, body=self.body)
        except:
            print("pacing - Failed!")

    def acs_register(self, cfgObj):
        """calls Agent configurations server to set up this agent"""
        url = "http://" + cfgObj["Ip"]
        url = url + ":" + str(cfgObj["Port"])
        url = url + "/v1/agents/my_http_config/config"
        data = json.dumps(cfgObj["Config"])
        # send request to ACS
        try:
            print("ACS reg'ing: " + data)
            self.http_client.fetch(url, callback=handle_async_request, method='POST', headers=self.headers, body=data)
        except:
            print("ACS registration failed")


# ----- test function

def tornado_bidder_run():
    """runs httpapi bidder agent"""

    # bind tcp port to launch processes on requests
    sockets = netutil.bind_sockets(CONFIG_OBJ["Bidder"]["Port"])

    # fork working processes
    process.fork_processes(0)

    # Tornado app implementation
    app = Application([url(r"/", TornadoFixPriceBidAgentRequestHandler)])

    # start http servers and attach the web app to it
    server = httpserver.HTTPServer(app)
    server.add_sockets(sockets)

    # perform following actions only in the parent process
    process_counter = process.task_id()
    if (process_counter == 0):
        # run dummy ad server
        adserver_win = Application([url(r"/", TornadoDummyRequestHandler)])
        winport = CONFIG_OBJ["Bidder"]["Win"]
        adserver_win.listen(winport)
        adserver_evt = Application([url(r"/", TornadoDummyRequestHandler)])
        evtport = CONFIG_OBJ["Bidder"]["Event"]
        adserver_evt.listen(evtport)

        # --instantiate budget pacer
        pacer = BudgetControl()
        pacer.start(CONFIG_OBJ)

        # add periodic event to call pacer
        PeriodicCallback(pacer.http_request, CONFIG_OBJ["Banker"]["Period"]).start()

    # main io loop. it will loop waiting for requests
    IOLoop.instance().start()


# run test of this module
if __name__ == '__main__':
    tornado_bidder_run()
