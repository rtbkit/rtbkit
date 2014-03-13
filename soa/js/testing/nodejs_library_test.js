// test 1

var mod1 = require('testlib1');
var mod2 = require('testlib2');
var mod3 = require('js2json_module');
var assert = require('assert');

var vows    = require('vows');

var tests = {
    'test1': {
        topic: true,
        mod1 :  function()
        {
            assert.equal(mod1.x, 10);
        },
        mod2: function()
        {
            assert.equal(mod2.y, 30);
        }
    }
};

vows.describe('nodejs_library_test').addVows(tests).export(module);

