/* fbx_parsing.h                                               -*- C++ -*-
   Jean-Sebastien Bejeau, 19 June 2013

   Code to parse FBX bid requests.
*/

#pragma once

#include "soa/types/value_description.h"
#include "soa/types/basic_value_descriptions.h"
#include "soa/types/json_parsing.h"
#include "fbx.h"
#include <boost/lexical_cast.hpp>

namespace Datacratic {

template<>
struct DefaultDescription<FBX::BidRequest>
    : public StructureDescription<FBX::BidRequest> {
    DefaultDescription();
};

template<>
struct DefaultDescription<FBX::RtbUserContext>
    : public StructureDescription<FBX::RtbUserContext> {
    DefaultDescription();
};

template<>
struct DefaultDescription<FBX::RtbPageContext>
    : public StructureDescription<FBX::RtbPageContext> {
    DefaultDescription();
};

template<>
struct DefaultDescription<FBX::BidResponse>
    : public StructureDescription<FBX::BidResponse> {
    DefaultDescription();
};

template<>
struct DefaultDescription<FBX::RtbBid>
    : public StructureDescription<FBX::RtbBid> {
    DefaultDescription();
};

template<>
struct DefaultDescription<FBX::RtbBidDynamicCreativeSpec>
    : public StructureDescription<FBX::RtbBidDynamicCreativeSpec> {
    DefaultDescription();
};



} // namespace Datacratic
