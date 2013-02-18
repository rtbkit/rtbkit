var sync = require('sync').makeSynchronous();

//var univ    = require('universe');

var sys     = require('sys');
var assert  = require('assert');
var vows    = require('vows');
var fs      = require('fs');
var mod     = require("signal_slot_test_module");

var tests = {
    'new handler': {
        topic: function () { return new mod.SignalSlotTest(); },
        'conversion':  function (t)
        {
            var c = t.checkCast();
            assert.equal(c[0], c[1]);
        },
        'cppType':     function (t)
        {
            assert.equal(t.cppType(), "SignalSlotTest");
        },
        'signalNames': function (t)
        {
            assert.deepEqual(t.signalNames().sort(),
                             ['event1', 'event2']);
        },
        'signalTypes': function (t)
        {
            var info1 = t.signalInfo('event1');
            assert.match(info1.callbackType, /std::string/);
            assert.equal(info1.inherited, false);
            assert.equal(info1.name, "event1");
            assert.equal(info1.objectType, 'SignalSlotTest');
        },
        'add handler': {
            topic: function (t)
            {
                var x = [0];
                var d = t.on("event1", function (arg1) { x[0] = arg1; });
                sys.puts(d);
                return { t: t, x: x, d: d };
            },
            'disconnector': function(info)
            {
                assert.equal(info.d.cppType(), "Datacratic::Slot");
            },
            'event1' : function (info) {
                var t = info.t;
                var x = info.x;

                t.event1(1);

                assert.deepEqual(x, [ 1 ]);

                t.event1(2);

                assert.deepEqual(x, [ 2 ]);

                var d = info.d;

                // Disconnect it
                d.call();

                t.event1(3);

                // Make sure it was disconnected
                assert.deepEqual(x, [ 2 ]);

                // Check that double disconnecting doesn't crash
                d.call();
                t.event1(3);
                assert.deepEqual(x, [ 2 ]);
            },
            'exception passing' : function (info) {
                var t = info.t;
                var x = info.x;

                function onEvent1(arg1)
                {
                    throw new Error("this is an error");
                }

                var d = t.on("event1", onEvent1);

                assert.throws(function () { t.event1(1); }, Error);

                // Disconnect it
                d();

                // Check that it doesn't throw
                t.event1(1);
            },
            'accurate exception passing' : function (info) {
                var t = new mod.SignalSlotTest();

                var times = 0;

                function onEvent1(arg1)
                {
                    times += 1;
                    if (times == 5)
                        throw new Error("this is an error");
                }

                t.on("event1", onEvent1);

                assert.throws(function() { t.event1(1, 10); }, Error);
                assert.equal(times, 5);
                assert.equal(t.lastIndex, 4);
            }
        }
    }
};

vows.describe('signal_slot_js_test').addVows(tests).export(module);
