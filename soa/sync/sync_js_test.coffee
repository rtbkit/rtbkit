vows = require 'vows'
assert = require 'assert'
testing_utils = require 'testing_utils'
sync = require 'sync'
beh_dist_store = require 'beh_dist_store'

fileManager = new testing_utils.FileManager

vows.describe('sync_js_test').export(module).addBatch
    'OutputStream':
        topic: -> 
            @path = fileManager.register 'sync_js_simple_write_test'
            new sync.OutputStream @path

        "simple log": (stream) ->
            nbLines = 2412
            nbItems = 2700
            for i in [0...nbLines]
                stream.log ([i].concat (0 for j in [0...nbItems])).join(' ')
            stream.close()
            reader = new beh_dist_store.FileReader @path
            n = 0 
            reader.forEachLine (nb, line) ->
                assert.equal line, ([nb].concat (0 for i in [0...nbItems])).join(' ')
                n++
                return true
            assert.equal n, nbLines
            
            
                



