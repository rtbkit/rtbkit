/** openrtb_bid_source.cc                                 -*- C++ -*-
    Eric Robert, 13 Aug 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

*/

#include "openrtb_bid_source.h"
#include "rtbkit/openrtb/openrtb_parsing.h"
#include "soa/service/http_header.h"
#include <mutex>

using namespace Datacratic;
using namespace RTBKIT;

namespace {
    class BidRequestReplayBuffer {
    public:
        BidRequestReplayBuffer() :
            cursor { 0 }
          , isFileLoaded { false }
        { }

        void loadFile(const std::string &fileName) {
            JML_TRACE_EXCEPTIONS(false)
            ML::filter_istream is(fileName);
            if (!is) {
                throw ML::Exception(ML::format("Could not load replay file: %s",
                                               fileName.c_str()));
            }

            std::cout << "Loading " << fileName << " replay file" << std::endl;

            size_t rejected, total;
            rejected = total = 0;
            // Assuming that everything in that file is openrtb 2.1
            auto p = OpenRTBBidRequestParser::openRTBBidRequestParserFactory("2.1");

            for (std::string line; getline(is, line); ) {
                try {
                    ++total;
                    auto br = p->parseBidRequest(line);
                    buffer.push_back(std::move(br));
                } catch (const ML::Exception &) {
                    ++rejected;
                }
            }

            std::cout << "Replay: parsed total of " << total << " lines, "
                      << rejected << " rejected (" << ((rejected * 100.0) / total)
                      << "%)" << std::endl;
            isFileLoaded = true;

        }

        OpenRTB::BidRequest next() {
            // Spinning until isFileLoaded is true
            while (!isFileLoaded) ; 

            size_t index = cursor.load();
            ExcCheck(index < buffer.size(), "replayCursor is invalid");

            auto br = buffer[index];
            size_t oldIndex { index };
            size_t newIndex;
            do {
                newIndex = (oldIndex + 1) % buffer.size();
            } while (!cursor.compare_exchange_weak(oldIndex, newIndex));

            return br;
        }

    private:
        std::atomic<size_t> cursor;
        std::atomic<bool> isFileLoaded;
        std::vector<OpenRTB::BidRequest> buffer;
    };

    BidRequestReplayBuffer replay;
    std::once_flag flag;
}

OpenRTBBidSource::
OpenRTBBidSource(Json::Value const & json) :
    BidSource(json),
    host(json.get("host", ML::hostname()).asString()),
    verb(json.get("verb", "POST").asString()),
    resource(json.get("resource", "/").asString()),
    p(OpenRTBBidRequestParser::openRTBBidRequestParserFactory("2.1")),
    replayFile(false)
{

    if (json.isMember("replayFile")) {
        replayFile = true;
        // Make sure we load the file only once
        std::call_once(flag, [&]() {
            replay.loadFile(json["replayFile"].asString());
        });
    }
}


OpenRTB::BidRequest
OpenRTBBidSource::
generateRequest()
{
    key += rng.random();

    OpenRTB::BidRequest req;
    req.id = Id(rng.random());
    req.tmax.val = 50;
    req.at = AuctionType::SECOND_PRICE;
    req.imp.emplace_back();
    auto & imp = req.imp[0];
    imp.id = Id(key);
    imp.banner.reset(new OpenRTB::Banner);
    imp.banner->w.push_back(300);
    imp.banner->h.push_back(250);
    req.user.reset(new OpenRTB::User);
    req.user->id = Id(rng.random());
    req.user->buyeruid = Id(rng.random());

    return req;
}

BidRequest OpenRTBBidSource::generateRandomBidRequest() {
    OpenRTB::BidRequest req;

    if (replayFile) {
        req = replay.next();
        req.id = Id(rng.random());
        for (auto &imp: req.imp) {
            key += rng.random();
            imp.id = Id(key);
        }
    }
    else {
        req = generateRequest();
    }

    StructuredJsonPrintingContext context;
    DefaultDescription<OpenRTB::BidRequest> desc;
    desc.printJson(&req, context);
    const std::string & content = context.output.toString();

    int length = content.length();

    std::string message = ML::format(
        "%s %s HTTP/1.1\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "accept: */*\r\n"
        "connection: Keep-Alive\r\n"
        "host: %s\r\n"
        "user-agent: be2/1.0\r\n"
        "x-openrtb-version: 2.1\r\n"
        "\r\n%s",
        verb.c_str(), resource.c_str(), length, host.c_str(), content.c_str());

    write(message);
    std::unique_ptr<BidRequest> br(p->parseBidRequest(content,
                                                      "openrtb",
                                                      "openrtb"));
    return BidRequest(*br);
}

auto
OpenRTBBidSource::
parseResponse(const std::string& rawResponse) -> std::pair<bool, std::vector<Bid>> {
    OpenRTB::BidResponse response;

    if(rawResponse.empty() || rawResponse.find("204 No Content") != std::string::npos ) {
        return std::make_pair(false, std::vector<Bid>());
    }

    try {
        HttpHeader header;

        header.parse(rawResponse);
        if (!header.contentLength || header.resource != "200") {
            //std::cerr << rawResponse << std::endl;
            return std::make_pair(false, std::vector<Bid>());
        }

        ML::Parse_Context context("payload", header.knownData.c_str(), header.knownData.size());
        StreamingJsonParsingContext json(context);
        DefaultDescription<OpenRTB::BidResponse> desc;
        desc.parseJson(&response, json);
    }
    catch (const std::exception & exc) {
        std::cerr << "invalid response received: " << exc.what() << std::endl;
        return std::make_pair(false, std::vector<Bid>());
    }

    std::vector<Bid> bids;

    for(auto i = 0; i != response.seatbid[0].bid.size(); ++i) {
        Bid bid;
        bid.adSpotId = response.seatbid[0].bid[i].impid;
        bid.maxPrice = USD_CPM(response.seatbid[0].bid[i].price.val);
        bids.push_back(bid);
    }

    return std::make_pair(true, bids);
}

namespace {

struct AtInit {
    AtInit()
    {
        PluginInterface<BidSource>::registerPlugin("openrtb",
						   [](Json::Value const & json) {
            return new OpenRTBBidSource(json);
        });
    }
} atInit;

}

