/* openrtb_parsing.h                                               -*- C++ -*-
   Mark Weiss, 28 March 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Code to parse AppNexus bid requests.
*/

#pragma once

#include <string>
#include "appnexus.h"
#include "rtbkit/openrtb/openrtb_parsing.h"
#include <type_traits>

// using namespace OpenRTB;
using std::string;


namespace Datacratic {


template<>
struct DefaultDescription<AppNexus::AdPosition>
    : public TaggedEnumDescription<AppNexus::AdPosition> {

    DefaultDescription()
    {
    }

    void parseJsonTyped(AppNexus::AdPosition * val,
                   JsonParsingContext & context) const
    {
        string appNexAdPos = context.expectStringAscii();

        if (appNexAdPos == "unknown") {
          val->val = AppNexus::AdPosition::UNKNOWN;
        } 
        else if (appNexAdPos == "above") {
          val->val = AppNexus::AdPosition::ABOVE;
        } 
        else if (appNexAdPos == "below") {
          val->val = AppNexus::AdPosition::BELOW;
        }
        else { // AN only supports the above three AdPos types.
               // ORTB supports others but AN does not.
          val->val = AppNexus::AdPosition::UNSPECIFIED;
        }
    }
};

template<>
struct DefaultDescription<AppNexus::BidRequestMsg>
    : public StructureDescription<AppNexus::BidRequestMsg> {
    DefaultDescription();
};

template<>
struct DefaultDescription<AppNexus::BidRequest>
    : public StructureDescription<AppNexus::BidRequest> {
    DefaultDescription();
};

template<>
struct DefaultDescription<AppNexus::BidInfo>
    : public StructureDescription<AppNexus::BidInfo> {
    DefaultDescription();
};

template<>
struct DefaultDescription<AppNexus::Segment>
    : public StructureDescription<AppNexus::Segment> {
    DefaultDescription();
};

template<>
struct DefaultDescription<AppNexus::InventoryAudit>
    : public StructureDescription<AppNexus::InventoryAudit> {
    DefaultDescription();
};

template<>
struct DefaultDescription<AppNexus::Tag>
    : public StructureDescription<AppNexus::Tag> {
    DefaultDescription();
};

template<>
struct DefaultDescription<AppNexus::Member>
    : public StructureDescription<AppNexus::Member> {
    DefaultDescription();
};

static_assert(std::is_same<typename GetDefaultDescriptionType<AppNexus::AdPosition>::type, DefaultDescription<AppNexus::AdPosition> >::type(), "wrong appnexus type");

} // namespace Datacratic
