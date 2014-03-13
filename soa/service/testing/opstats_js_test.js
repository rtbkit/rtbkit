/* opstats_js_test.js
 * Jeremy Barnes, 5 August 2011
 * Copyright (c) 2011 Datacratic.  All rights reserved.
 */

var CarbonConnector = require('opstats').CarbonConnector;

var cc = new CarbonConnector('127.0.0.1:2003', 'test.opstats');

cc.recordLevel("level", 10);

cc.dump();

cc.stop();


