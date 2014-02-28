var rtb = require("rtb");
var util = require("util");
var BidRequest = require('bid_request').BidRequest;
require('sync').makeSynchronous();

var router = new rtb.Router();
router.start();

console.log("stats: " + util.inspect(router.getStats()));

var intervalid, watchdogIntervalid;

var numFinished = 0;

function onAuctionFinished(auction)
{
    console.log("auction finished");
    console.log(auction);
    console.log(this);

    ++numFinished;
    if (numFinished < 2)
        return;

    //throw new Error("hello");

    clearInterval(intervalid);
    clearInterval(watchdogIntervalId);

    console.log("shutting down");

    router.shutdown();

    console.log("done shutting down baby");

}

var request = {
    id: "hello",
    timestamp: +(new Date()),
    userIds: { "provider": Math.random().toFixed(3) },
    url:"www.sitename.com"
};

router.injectAuction(onAuctionFinished, new BidRequest(JSON.stringify(request), "datacratic"));

var request2 = {
    "id" : "76543210-0000-1234-1234-123456789abc",
    "url":"http://www.xyzxyz.com/calendar",
    "isTest":false,
    meta: {
        "original_br": {
            "close":null,
            "cppType":null,
            "data": {
                "anonymous_id":"",
                "browser":"Firefox",
                "city":"Toronto",
                "country":"CA",
                "detected_language":"en",
                "guid":"12321321321312",
                "host":"www.xyzxyz.com",
                "os":"Windows NT 6",
                "region":"CA-ON",
                "req_id":"76543210-0000-1234-1234-123456789abc",
                "url":"http://www.xyzxyz.com/calendar",
                "verticals": [
                    {"verticalid":358,"weight":0.3334270},
                    {"verticalid":1187,"weight":0.2664740},
                    {"verticalid":359,"weight":0.2604160},
                    {"verticalid":3,"weight":0.1396830}
                ]
            },
            "date":1302101665.9640,
            "id":2421817,
            "type":"request_type",
            "wrapperType":null
        }
    },
    "spots":[
        {
            "id":"4d9c7ea2-0000-cce1-0a2a-cd1222793d86-0",
            "reservePrice":0,
            banner: {"h":90, "w":728 }
        }
    ],
    "timestamp":1302101665.9640,
    "userIds": { "provider": "123421342umbpqojTJCqf1kqLFzXJ0" }
};

router.injectAuction(onAuctionFinished, new BidRequest(JSON.stringify(request2), "datacratic"));

function onTick()
{
    console.log(router.numAuctionsInProgress() + " auctions in progress");
    console.log(router.getStats());
}

function onTooLong()
{
    console.log("WATCHDOG: test took too long");
    process.exit(1);
}

intervalid = setInterval(onTick, 2000);

watchdogIntervalId = setInterval(onTooLong, 10000);