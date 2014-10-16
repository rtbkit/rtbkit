var mod     = require('js2json_module');
var sys     = require('sys');
var assert  = require('assert');
var vows    = require('vows');

var tests = {
    'test_fromto': {
        topic: new mod.FromTo,
        roundTrips :  function(x) {
            var samples = [
                           true,
                           1,
                           1.2,
                           "a",
                           [1,2],
                           {a:1},
                           {},[],
                           [{a:1}, {b:"1"}],
                           {a:true, b:1, c: "1", d: [1,2,true,false, {a:2.2}], e: {f:"g"}}];
            samples.forEach(function(sample){
                assert.equal(JSON.stringify(sample),
                        JSON.stringify(x.roundTrip(sample)));
            });
        },
        dates :  function(x) {
            var obj = {a: new Date()};

            assert.equal(x.roundTrip(obj).a, obj.a.getTime()/1000);
        },
        fromWithin :  function(x) {
            var obj = {a: 1};
            assert.equal(JSON.stringify(obj),
                    JSON.stringify(x.getJSON1()));
        },
        fromWithin2 :  function(x) {
            var obj = {a: {b : [1,2.2], c: true}, d: "string"};
            assert.equal(JSON.stringify(obj),
                    JSON.stringify(x.getJSON2()));
            assert.equal(JSON.stringify(obj),
                    JSON.stringify(x.roundTrip(x.getJSON2())));
        },
        testBigInt: function (x) {
            assert.equal(x.getLongLongInt(),
                         8589934592);
        },
        testBigIntViaJsonValue: function (x) {
            /*
             * "long_long" is extracted to avoid this magic failure:
             *  » expected { long_long: 8589934592 },
                  got      { long_long: 8589934592 } (==)
             */
            assert.equal(x.getJSONLongLongInt()["long_long"],
                         8589934592);
        }
    },
};


vows.describe('js2json_test').addVows(tests).export(module);
