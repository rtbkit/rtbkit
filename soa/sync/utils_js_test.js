var utils  = require('sync_utils');
var sync   = require('sync');
var assert = require("assert");
var sys    = require("sys");

// test various forEachLine things

function onEachLine()
{
    assert.equal(true, false);
}

var numDone = 0;

assert.throws(function () { utils.forEachLine("nonExistantFilename", onEachLine); });

assert.throws(function () { utils.forEachLine("platform/utils/utils_js_test.js", 1); });

assert.equal(numDone, 0);


// Test output streams

console.log(sys.inspect(utils));
console.log(sys.inspect(this));
console.log(sys.inspect(cout));

n = 1000
for (var i = 0;  i < n;  ++i)
    cout.log("hello", i);

for (i = 0;  i < n;  ++i)
    cerr.log("hello2", i);

n = 100000
for (var i = 0;  i < n;  ++i)
    console.log("hello", i);

assert.equal(utils.runCmd('echo "patate"'), "patate\n");
//assert.notEqual(utils.runCmd('+++patate poil+++1432'), 0);
