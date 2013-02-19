vows        = require "vows"
assert      = require "assert"
bid_request = require "bid_request"
fs          = require "fs"

brfixture = "MATCHEDWIN\t2012-May-06 14:11:02.87818\t8b703eb9-5ebf-4001-15e8-10d6000003a0\t0\taero_23_probe_43-agent-8450\taero_23_probe_43\t2170\t5172994\t5172994.500000\t{\"!!CV\":\"0.1\",\"exchange\":\"casale\",\"id\":\"8b703eb9-5ebf-4001-15e8-10d6000003a0\",\"ipAddress\":\"76.11.50.50\",\"language\":\"en\",\"location\":{\"cityName\":\"Grande Prairie\",\"countryCode\":\"CA\",\"dma\":0,\"postalCode\":\"0\",\"regionCode\":\"AB\",\"timezoneOffsetMinutes\":240},\"protocolVersion\":\"0.3\",\"provider\":\"AdGear\",\"segments\":{\"adgear\":null},\"spots\":[{\"formats\":[\"160x600\"],\"id\":\"22202919\",\"position\":\"NONE\",\"reservePrice\":0}],\"timestamp\":1336313462.550589,\"url\":\"http://emedtv.com/search.html?searchString=•skin\",\"userAgent\":\"Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)\",\"userIds\":{\"ag\":\"7d837be6-94de-11e1-841f-68b599c88614\",\"casl\":\"PV08FEPS1KQAAGcrbUsAAABY\",\"prov\":\"7d837be6-94de-11e1-841f-68b599c88614\",\"xchg\":\"PV08FEPS1KQAAGcrbUsAAABY\"}}\t45221\tnetProphets / Run Of Network (RON)\taero_23_probe_43";


tests =
    "Parse UTF8 bid request":
        topic: ->
                data = brfixture;
                brdata = data.split('\t')[9]
                br = new bid_request.BidRequest brdata, "datacratic"
                this.callback(null, br)

        "test can deal with invalid chars in url": (br) ->
            assert.equal typeof(br), "object"
            assert.equal br.url, 'http://emedtv.com/search.html?searchString=%E2%80%A2skin'

        "check cant create from undefined": ->
            assert.throws -> new bid_request.BidRequest(undefined)

        "check if all the fields are extracted": (br) ->
            assert.equal br.userAgent, "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)"


vows.describe('simulator_test').export(module).addVows(tests)
