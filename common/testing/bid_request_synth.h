/** bid_request_synth.h                                 -*- C++ -*-
    Rémi Attab, 25 May 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Bid request synthetizer.

*/

#pragma once

#include <memory>
#include <istream>
#include <ostream>

namespace Json { struct Value; }

namespace RTBKIT {

namespace Synth { struct Node; }

/******************************************************************************/
/* BID REQUEST SYNTH                                                          */
/******************************************************************************/

struct BidRequestSynth
{
    BidRequestSynth();

    void record(const Json::Value& json);
    Json::Value generate() const;

    void dump(std::ostream& stream);
    void load(std::istream& stream);

private:
    std::shared_ptr<Synth::Node> values;
};


} // namespace RTBKIT
