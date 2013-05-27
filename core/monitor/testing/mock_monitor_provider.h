/** mock_monitor_provider.h                                 -*- C++ -*-
    Rémi Attab, 24 May 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.
*/

#pragma once

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
    MockMonitorProvider(const std::string& providerClass)
        : providerClass_(providerClass),
          providerName_("mock-provider"),
          status_(false),
          delay_(0)
    {
    }

    std::string getProviderClass()
        const
    {
        return providerClass_;
    }

    MonitorIndicator getProviderIndicators()
        const
    {
        using namespace std;
        if (delay_) {
            cerr << ML::format("%s: sleeping for %d seconds\n",
                               CURRENT_METHOD(MockMonitorProvider),
                               delay_);
            ML::sleep(delay_);
        }

        MonitorIndicator ind;

        ind.serviceName = providerName_;
        ind.status = status_;
        cerr << ML::format("%s: returning %s\n",
                           CURRENT_METHOD(MockMonitorProvider),
                           boost::trim_copy(ind.toJson().toString()));

        return ind;
    }

    std::string providerClass_;
    std::string providerName_;
    bool status_;
    int delay_;
};

}
