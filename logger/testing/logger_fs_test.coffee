fs = require "fs"
util = require "util"
assert = require "assert"
vows = require "vows"

logger = require "logger"


createTestFolder = (name, mode) ->
    folderPath = "./logtest-#{name}"

    try fs.rmdirSync(folderPath)
    try fs.mkdirSync(folderPath, mode)

    console.log "created ", folderPath

    return folderPath

rotFolderAttempts = 0
rotFileAttempts = 0

goodFolder = ""
badFolder = ""
logfile = "/a"

#goodFolder = createTestFolder("perm-good", "0764")
#badFolder = createTestFolder("perm-bad", "0000")

vows.describe('logger_permision_test').export(module).addVows
    "FileOutput permission tests":
        topic: ->
            goodFolder = createTestFolder("perm-good", "0764")
            badFolder = createTestFolder("perm-bad", "0000")

        "Exception on construction": ->
            assert.throws -> new logger.FileOutput(badFolder + logfile)

        "Exception on open": ->
            l = new logger.FileOutput()
            assert.throws -> l.open(badFolder + logfile)

        "Exception on rotate": ->
            console.log goodFolder
            l = new logger.FileOutput(goodFolder + logfile)
            assert.throws -> l.rotate(badFolder + logfile)

        teardown: ->
            fs.unlink(goodFolder + logfile)
            fs.rmdir(goodFolder)
            fs.rmdir(badFolder)


        # Causes an exception to be thrown in another thread which isn't caught.
        # This results in the unexpected handler to be called which kills the whole process.

        # "rot_folder":
        #         topic: ->
        #                 @folderPath = createTestFolder("rd", "0000")
        #                 try
        #                         rotFolderLog = new logger.RotatingFileOutput()
        #                 catch err
        #                         console.log "Coffee.rot_folder Fail."
        #                         throw err

        #                 rotFolderLog.onBeforeLogRotation = (str) -> @callback("before",str)
        #                 rotFolderLog.onAfterLogRotation = (str) -> @callback("after",str)

        #                 try
        #                         rotFolderLog.open(@folderPath + "/%s/b", "1s")
        #                 catch err
        #                         console.log "Coffee.rotFolderLog.open Fail"
        #                         throw err

        #         "Before rotation counter": (type, str) ->
        #                 return unless type == "before"
        #                 rotFolderAttempts++
        #                 assert.ok rotFolderAttempts <= 1, "Attempted to rotate after an error."

        #         "Exception on rotation": (type, str) ->
        #                 return unless type == "after"
        #                 assert.ok false, "Continued to execute even after the rotation failed."

        #         teardown: -> fs.rmdir(@folderPath)

