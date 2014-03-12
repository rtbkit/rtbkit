vows        = require "vows"
assert      = require "assert"
bid_request = require "bid_request"
fs          = require "fs"
util        = require "util"

brfixture = "MATCHEDWIN\t2012-May-06 14:11:02.87818\t8b703eb9-5ebf-4001-15e8-10d6000003a0\t0\taero_23_probe_43-agent-8450\taero_23_probe_43\t2170\t5172994\t5172994.500000\t{\"!!CV\":\"0.1\",\"exchange\":\"abcd\",\"id\":\"8b703eb9-5ebf-4001-15e8-10d6000003a0\",\"ipAddress\":\"76.aa.xx.yy\",\"language\":\"en\",\"location\":{\"cityName\":\"Grande Prairie\",\"countryCode\":\"CA\",\"dma\":0,\"postalCode\":\"0\",\"regionCode\":\"AB\",\"timezoneOffsetMinutes\":240},\"protocolVersion\":\"0.3\",\"provider\":\"xxx1\",\"segments\":{\"xxx1\":null},\"imp\":[{\"formats\":[\"160x600\"],\"id\":\"22202919\",\"position\":\"NONE\",\"reservePrice\":0}],\"timestamp\":1336313462.550589,\"url\":\"http://emedtv.com/search.html?searchString=•skin\",\"userAgent\":\"Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)\",\"userIds\":{\"ag\":\"7d837be6-94de-11e1-841f-68b599c88614\",\"abcd\":\"PV08FEPS1KQAAGcrbUsAAABY\",\"prov\":\"7d837be6-94de-11e1-841f-68b599c88614\",\"xchg\":\"PV08FEPS1KQAAGcrbUsAAABY\"}}\t45221\tcampaign name";

console.log assert

tests =
    "Parse UTF8 bid request":
        topic: ->
                data = brfixture;
                brdata = data.split('\t')[9]
                br = new bid_request.BidRequest brdata, "datacratic"
                this.callback(null, br)

        "test can deal with invalid chars in url": (br) ->
            console.log util.inspect br
            assert.equal typeof(br), "object"
            assert.equal br.url, 'http://emedtv.com/search.html?searchString=%E2%80%A2skin'

        "check cant create from undefined": ->
            assert.throws -> new bid_request.BidRequest(undefined)

        "check if all the fields are extracted": (br) ->
            assert.equal br.userAgent, "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)"

    "test programmatic access to all fields":
        topic: ->
            br = new bid_request.BidRequest()
            this.callback(null, br)

        "test can set id": (br) ->
            br.id = "hello"
            assert.equal br.id, "hello"

        "test can set location fields": (br) ->
            br.location.cityName = "Texas"
            assert.equal br.location.cityName, "Texas"

        "test can set imp": (br) ->
            #br.imp = [ {} ]

            br.imp = [ new bid_request.AdSpot() ]
            assert.equal br.imp.length, 1
            console.log br.imp[0]
            console.log br.imp[0].id
            br.imp[0].id = "hello2";
            console.log br.imp[0]
            assert.equal br.imp[0].id, "hello2"
            console.log 'setting banner -------'
            br.imp[0].banner = { w: 10, h: 10}
            console.log 'finished setting banner -------'
            console.log br.imp[0]
            assert.isNotNull br.imp[0].banner
            assert.equal br.imp[0].banner.w, 10
            assert.equal br.imp[0].banner.h, 10

        "test can set imp from object literal": (br) ->
            br.imp = [ {id:123456, banner: { w: 20, h: 20 } } ]
            assert.equal br.imp.length, 1
            assert.equal br.imp[0].id, 123456
            assert.equal br.imp[0].banner.w, 20
            assert.equal br.imp[0].banner.h, 20
            #assert.deepEqual br.imp[0].formats, ["20x20"]


vows.describe('bid_request_test').export(module).addVows(tests)
