var mod     = require('v8_inheritance_module');
var sys     = require('sys');
var assert  = require('assert');
var vows    = require('vows');

var tests = {
    'test_base': {
        topic: new mod.Base,
        number :  function(b) { assert.equal(b.number(), 27); },
        number2 : function(b) { assert.equal(b.number2(), 27); },
        type:     function(b) { assert.equal(b.type(), "Base"); },
        otherType:function(b) {
            var b2 = new mod.Base;
            assert.equal(b.otherType(b2), "Base");
            assert.equal(b2.otherType(b), "Base");
            assert.throws(function() { b.otherType(undefined); });
            assert.throws(function() { b.otherType({}); });
            assert.throws(function() { b.otherType(new mod.Base2); });
        }
    },

    'test_derived': {
        topic: new mod.Derived,
        number :  function(b) { assert.equal(b.number(), 37); },
        number2 : function(b) { assert.equal(b.number2(), 27); },
        type:     function(b) { assert.equal(b.type(), "Derived"); },
        otherType:function(b) {
            var b2 = new mod.Base;
            assert.equal(b.otherType(b2), "Base");
            assert.equal(b2.otherType(b), "Derived");
            assert.equal(b.otherTypeDerived(b), "Derived");
            assert.throws(function() { b.otherTypeDerived(new mod.Base); });
        }

    },

    'test_rederived': {
        topic: new mod.ReDerived,
        number :  function(b) { assert.equal(b.number(), 47); },
        number2 : function(b) { assert.equal(b.number2(), 27); },
        type:     function(b) { assert.equal(b.type(), "ReDerived"); },
        otherType:function(b) {
            var b2 = new mod.Base;
            assert.equal(b.otherType(b2), "Base");
            assert.equal(b2.otherType(b), "ReDerived");
        }
    }
};


vows.describe('v8_inheritance_test').addVows(tests).export(module);
