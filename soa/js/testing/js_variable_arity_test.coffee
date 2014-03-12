vows   = require "vows"
assert = require "assert"
fs     = require "fs"
util   = require "util"
atm    = require "js_variable_arity_module"

vows.describe('js_variable_arity_test').export(module).addVows

    "arity_test":
        topic: ->
            return new atm.ArityTestClass

        "param_test": (topic) ->
            assert.equal topic.method(1), 1
            assert.equal topic.method(10), 10
            assert.equal topic.method(), 10

        teardown: (topic) ->
