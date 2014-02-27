var mod     = require('js_exception_passing_module');
var sys     = require('sys');
var assert  = require('assert');
var vows    = require('vows');

var tests = {
    'test_base': {
        topic: new mod.TestException,
        mlException :  function(e) {
            try {
                e.testMlException();
                assert(false);
            } catch (e) {
                assert.match(e.message, /hello/);
            }
        },
        mlException2:  function(e) {
            try {
                e.testMlException2();
                assert(false);
            } catch (e) {
                assert.match(e.message, /hello2/);
            }
        },
        stdException: function(e) {
            try {
                e.testStdException();
                assert(false);
            } catch (e) {
                assert.match(e.message, /logic_error/);
                assert.match(e.message, /bad medicine/);
            }
        },
        passThrough: function(e) {
            try {
                e.testPassThrough(function() {throw Error("pass through"); });
                assert(false);
            } catch (e) {
                assert.match(e.message, /pass through/);
            }
        }

    },
    'test_backtrace_c++': {
        topic: new mod.TestException,
        isDatacratic : function (e) {
            try {
                e.testStdException();
            } catch (e) {
                //sys.puts(sys.inspect(e));
                //sys.puts(sys.inspect(e.stack));
                //assert.equal(e.recoset, "hello");
                assert.match(e.stack, /\[C\+\+\] TestExceptionJS/);
            }
        }
    }
};

vows.describe('js_exception_passing_module').addVows(tests).export(module);

//var exc = new mod.TestException;
//exc.testStdException();
