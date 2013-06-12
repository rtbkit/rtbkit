assert    = require "assert"
vows      = require "vows"
i_logger_metrics = require "iloggermetricscpp"
disableAll = 0

sortFct=(a, b) -> a > b

vows.describe("Attributor JS Test").export(module).addVows
    "Mama Test":
        topic: ->
            return ""
            

        "Trivial test": (fun) ->
            i_logger_metrics.logMetrics("coco", "est", "beau", 10)
            i_logger_metrics.logMeta({"key" : "Value"})
