/* rtb_bid_request_segments_test.js
 * Wolfgang Sourdeau, April 2013
 * Copyright (c) 2013 Datacratic.  All rights reserved.
 *
 * Tests for the bid request.
 */

var vows   = require('vows');
var assert = require('assert');
var brm    = require('bid_request');

var requestWithSegments
    = {"!!CV": "RTBKIT-JSON-1.0",
       "bidCurrency": ["USD"],
       "device": {"carrier": "Carrier Something",
                  "geo": {"city": "Chicago",
                          "country": "US",
                          "metro": "312",
                          "region": "IL"},
                  "ip": "666.21.123.234",
                  "language": "en-US, en;q=0.8",
                  "make": "Motorola",
                  "model": "DROID RAZR",
                  "os": "Android",
                  "osv": "2.3.5",
                  "ua": "Mozilla/5.0 (Linux; Android 4.1.2; DROID RAZR Build/9.8.2O-72_VZW-16) AppleWebKit/537.31 (KHTML, like Gecko) Chrome/26.0.1410.58 Mobile Safari/537.31"},
       "exchange": "zeExchange",
       "id": "148ce45e066a4ef2e6e4b927dc881ad6f209cf4b",
       "imp": [{"banner": {"h": 250,
                           "pos": 3,
                           "w": 300},
                "formats": ["300x250"],
                "id": "1"}],
       "ipAddress": "666.21.123.234",
       "language": "en-US,en;q=0.8",
       "location": {"cityName": "Chicago",
                    "countryCode": "US",
                    "regionCode": "IL"},
       "provider": "zeProvider",
       "segments": {"100": [1, 43, 125],
                    "segid": ["Atext", "text1"],
                    "tags": ["cid: 54455"]},
       "site": {"domain": "http://domain.com",
                "id": "1-18039",
                "page": "http://domain.com/",
                "publisher": {"id": "9725"}},
       "spots": [{"banner": {"h": 250,
                             "pos": 3,
                             "w": 300},
                  "formats": ["300x250"],
                  "id": "1"}],
       "timestamp": 1366768734.7632546,
       "url": "http:  //domain.com/",
       "user": {"buyeruid": "somid",
                "data": [{"id": "100",
                          "segment": [{"id": "1"},
                                      {"id": "43"},
                                      {"id": "125"}]},
                         {"id": "segid",
                          "segment": [{"id": "text1"},
                                      {"id": "Atext"}]}],
                "geo": {},
                "id": "localid"},
       "userAgent": "Mozilla/5.0 (Linux; Android 4.1.2; DROID RAZR Build/9.8.2O-72_VZW-16) AppleWebKit/537.31 (KHTML, like Gecko) Chrome/26.0.1410.58 Mobile Safari/537.31",
       "userIds": {"prov": "somid",
                   "xchg": "localid"}};

var segmentsTest = {
    topic: function() {
        return new brm.BidRequest(JSON.stringify(requestWithSegments),
                                  "datacratic");
    },
    checkSegments: function(x) {
        var segments = x.segments;
        var result = {};
        assert("100" in segments,
               "'100' must be present in segment list");
        assert("segid" in segments,
               "'segid' must be present in segment list");

        var seg0 = x.segments["BLA"];
        assert(seg0 == undefined, "bad1");

        var seg100 = x.segments["100"];
        assert(seg100 != undefined, "segment['100'] is undefined?");
        seg100 = x.segments[100];
        assert(seg100 != undefined, "segment[100] is undefined?");
        var seg100Values = {};

        for (var i = 0; i < seg100.toArray().length; i++) {
            seg100Values[seg100[i]] = true;
        }
        assert("1" in seg100Values,
               "'1' must be present in segments['100']");
        assert("43" in seg100Values,
               "'43' must be present in segments['100']");
        assert("125" in seg100Values,
               "'125' must be present in segments['100']");

        var segid = x.segments["segid"];
        assert(segid != undefined, "segments['segid'] is undefined?");
        var segidValues = {};
        for (var i = 0; i < segid.toArray().length; i++) {
            segidValues[segid[i]] = true;
        }
        assert("text1" in segidValues,
               "'text1' must be present in segments['segid']");
        assert("Atext" in segidValues,
               "'Atext' must be present in segments['segid']");
    }
};


var tests = {
    'segments': [ segmentsTest ]
};

vows.describe('rtb_bid_request_segments_test').addVows(tests).export(module);
