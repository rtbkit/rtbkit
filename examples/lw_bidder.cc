#include <iostream>
#include <thread>
#include <chrono>
#include <string>

#include "rtbkit/api/api.h"

using namespace std;

 auto agent_config = "{\"lossFormat\":\"lightweight\",\"winFormat\":\"full\",\"test\":false,\"minTimeAvailableMs\":5,\"account\":[\"hello\",\"world\"],\"bidProbability\":0.1000000014901161,\"creatives\":[{\"format\":\"728x90\",\"id\":2,\"name\":\"LeaderBoard\"},{\"format\":\"160x600\",\"id\":0,\"name\":\"LeaderBoard\"},{\"format\":\"300x250\",\"id\":1,\"name\":\"BigBox\"}],\"errorFormat\":\"lightweight\",\"externalId\":0}";
auto proxy_config = "{\"installation\":\"rtb-test\",\"location\":\"mtl\",\"zookeeper-uri\":\"localhost:2181\",\"portRanges\":{\"logs\":[16000,17000],\"router\":[17000,18000],\"augmentors\":[18000,19000],\"configuration\":[19000,20000],\"postAuctionLoop\":[20000,21000],\"postAuctionLoopAgents\":[21000,22000],\"banker.zmq\":[22000,23000],\"banker.http\":9985,\"agentConfiguration.zmq\":[23000,24000],\"agentConfiguration.http\":9986,\"monitor.zmq\":[24000,25000],\"monitor.http\":9987,\"adServer.logger\":[25000,26000]}}";


int main()
{
	RTBKIT::api::Bidder  bob("BOB", proxy_config);
    bob.init();
    bob.doConfig (agent_config);

    while (true) this_thread::sleep_for(chrono::seconds(10));

    // Won't ever reach this point but this is how you shutdown an agent.
    bob.shutdown();

    return 0;
}

