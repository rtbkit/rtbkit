/* rtb_new_format_test.js
 * Jeremy Barnes, 14 June 2011
 * Copyright (c) 2011 Datacratic.  All rights reserved.
 *
 * Check that we can parse the new bid request format.
 */

var utils = require('sync_utils');
var bid_request = require("bid_request");
var util = require("util");
var assert = require("assert");

var numDone = 0;
var passed = 0;

function forEachTestCase(line, lineNum)
{
    if (line == "") {
        ++passed;
        return;
    }
    console.log("running line");
    try {
        var request = new bid_request.BidRequest(line);
        ++passed;
    } catch (e) {
        console.log("error parsing line number " + lineNum);
        console.log(line);
        console.log(util.inspect(e));
    }
}

var numDone = utils.forEachLine("rtbkit/core/router/testing/new_protocol2-datacratic.log",
                                forEachTestCase);

assert.equal(numDone, passed);
