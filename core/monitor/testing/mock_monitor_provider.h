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
        : providerName_("mock-provider"), status_(false), delay_(0)
    {
    }

    std::string getProviderName()
        const
    {
        return providerName_;
    }

    Json::Value getProviderIndicators()
        const
    {
        using namespace std;
        if (delay_) {
            cerr << ML::format("%s: sleeping for %d seconds\n",
                               CURRENT_METHOD(MockMonitorProvider),
                               delay_);
            ML::sleep(delay_);
        }

        Json::Value value(Json::objectValue);

        value["status"] = (status_ ? "ok" : "failure");

        cerr << ML::format("%s: returning %s\n",
                           CURRENT_METHOD(MockMonitorProvider),
                           boost::trim_copy(value.toString()));

        return value;
    }

    std::string providerName_;
    bool status_;
    int delay_;
};

}
