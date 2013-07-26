/* openrtb_parsing.h                                               -*- C++ -*-
   Mark Weiss, 28 March 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Code to parse AppNexus bid requests.
*/

#pragma once

#include <string>
#include "appnexus.h"
#include "openrtb/openrtb_parsing.h"

// using namespace OpenRTB;
using std::string;


namespace Datacratic {

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


} // namespace Datacratic
