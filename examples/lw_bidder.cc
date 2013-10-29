#include <iostream>
#include <thread>
#include <chrono>
#include <string>

#include "rtbkit/api/api.h"
#include "rtbkit/common/bids.h"

using namespace std;
using namespace RTBKIT;

auto agent_config = "{\"lossFormat\":\"lightweight\",\"winFormat\":\"full\",\"test\":false,\"minTimeAvailableMs\":5,\"account\":[\"hello\",\"world\"],\"bidProbability\":0.1000000014901161,\"creatives\":[{\"format\":\"728x90\",\"id\":2,\"name\":\"LeaderBoard\"},{\"format\":\"160x600\",\"id\":0,\"name\":\"LeaderBoard\"},{\"format\":\"300x250\",\"id\":1,\"name\":\"BigBox\"}],\"errorFormat\":\"lightweight\",\"externalId\":0}";
auto proxy_config = "{\"installation\":\"rtb-test\",\"location\":\"mtl\",\"zookeeper-uri\":\"localhost:2181\",\"portRanges\":{\"logs\":[16000,17000],\"router\":[17000,18000],\"augmentors\":[18000,19000],\"configuration\":[19000,20000],\"postAuctionLoop\":[20000,21000],\"postAuctionLoopAgents\":[21000,22000],\"banker.zmq\":[22000,23000],\"banker.http\":9985,\"agentConfiguration.zmq\":[23000,24000],\"agentConfiguration.http\":9986,\"monitor.zmq\":[24000,25000],\"monitor.http\":9987,\"adServer.logger\":[25000,26000]}}";

int main()
{
    RTBKIT::api::Bidder  bob("BOB", proxy_config);
    bob.bid_request_cb_ = [&] (double               timestamp,
                               const std::string&   id,           // Auction i
                               const std::string&   bidRequest_str,
                               const std::string&   bids_str,
                               double               timeLeftMs,   // Time left of the bid reques
                               const std::string&   augmentations,
                               const std::string&   wcm) {
        auto bids = Bids::fromJson(bids_str);
        // auto br = RTBKIT::BidRequest::createFromJson(Json::parse(bidRequest_str));
        for (Bid& bid : bids) {

            // In our example, all our creatives are of the different sizes so
            // there should only ever be one biddable creative. Note that that
            // the router won't ask for bids on imp that don't have any
            // biddable creatives.
            ExcAssertEqual(bid.availableCreatives.size(), 1);
            int availableCreative = bid.availableCreatives.front();

            // We don't really need it here but this is how you can get the
            // AdSpot and Creative object from the indexes.
            // (void) br.imp[bid.spotIndex];
            //(void) config.creatives[availableCreative];

            // Create a 0.0001$ CPM bid with our available creative.
            // Note that by default, the bid price is set to 0 which indicates
            // that we don't wish to bid on the given spot.
            bid.bid(availableCreative, MicroUSD(123));
        }

        // A value that will be passed back to us when we receive the result of
        // our bid.
        Json::Value metadata = 42;

        // Send our bid back to the agent.
        bob.doBid(id, bids.toJson().toString());
    };
    bob.init();
    bob.doConfig (agent_config);
    bob.start (true);
    while (true) this_thread::sleep_for(chrono::seconds(10));

    // Won't ever reach this point but this is how you shutdown an agent.
    bob.shutdown();

    return 0;
}

