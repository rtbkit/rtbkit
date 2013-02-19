#include "boost/algorithm/string.hpp"
#include "jml/arch/format.h"
#include "soa/jsoncpp/value.h"
#include "rtbkit/core/monitor/monitor_provider.h"

#define CURRENT_METHOD(x) \
    ( std::string(#x) + "::" + std::string(__FUNCTION__) )

namespace RTBKIT {

struct MockMonitorProvider
    : public MonitorProvider
{
    MockMonitorProvider()
        : status(false), delay(0)
    {
    }

    Json::Value getMonitorIndicators()
    {
        using namespace std;
        if (delay) {
            cerr << ML::format("%s: sleeping for %d seconds\n",
                               CURRENT_METHOD(MockMonitorProvider),
                               delay);
            ML::sleep(delay);
        }

        Json::Value value(Json::objectValue);

        value["status"] = (status ? "ok" : "failure");

        cerr << ML::format("%s: returning %s\n",
                           CURRENT_METHOD(MockMonitorProvider),
                           boost::trim_copy(value.toString()));

        return value;
    }

    bool status;
    int delay;
};

}
