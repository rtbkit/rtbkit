/* openrtb_parsing.h                                               -*- C++ -*-
   Jeremy Barnes, 22 February 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Code to parse OpenRTB bid requests.
*/

#pragma once

#include "soa/types/value_description.h"
#include "soa/types/basic_value_descriptions.h"
#include "soa/types/json_parsing.h"
#include "openrtb.h"
#include <boost/lexical_cast.hpp>

namespace Datacratic {

template<>
struct DefaultDescription<OpenRTB::ContentCategory>
    : public ValueDescriptionI<OpenRTB::ContentCategory> {

    DefaultDescription()
    {
    }

    virtual void parseJsonTyped(OpenRTB::ContentCategory * val,
                                JsonParsingContext & context) const
    {
        val->val = context.expectStringAscii();
    }

    virtual void printJsonTyped(const OpenRTB::ContentCategory * val,
                                JsonPrintingContext & context) const
    {
        context.writeString(val->val);
    }
};

template<>
struct DefaultDescription<OpenRTB::MimeType>
    : public ValueDescriptionI<OpenRTB::MimeType, ValueKind::STRING> {

    virtual void parseJsonTyped(OpenRTB::MimeType * val,
                                JsonParsingContext & context) const
    {
        val->type = context.expectStringAscii();
    }

    virtual void printJsonTyped(const OpenRTB::MimeType * val,
                                JsonPrintingContext & context) const
    {
        context.writeString(val->type);
    }
};

#if 0
#define DECLARE_TAGGED_ENUM(Name)                   \
    TaggedEnumDescription<Name> *                   \
    getDefaultDescription(Name *)                   \
    {                                               \
        return new TaggedEnumDescription<Name>();   \
    }

DECLARE_TAGGED_ENUM(OpenRTB::VideoQuality);

template<>
struct DefaultDescription<OpenRTB::VideoQuality>
    : public TaggedEnumDescription<OpenRTB::VideoQuality> {
    DefaultDescription()
    {
    }
};

template<>
struct DefaultDescription<OpenRTB::AuctionType>
    : public TaggedEnumDescription<OpenRTB::AuctionType> {

    DefaultDescription()
    {
    }
};

template<>
struct DefaultDescription<OpenRTB::BannerAdType>
    : public TaggedEnumDescription<OpenRTB::BannerAdType> {

    DefaultDescription()
    {
    }
};

template<>
struct DefaultDescription<OpenRTB::CreativeAttribute>
    : public TaggedEnumDescription<OpenRTB::CreativeAttribute> {

    DefaultDescription()
    {
    }
};

template<>
struct DefaultDescription<OpenRTB::ApiFramework>
    : public TaggedEnumDescription<OpenRTB::ApiFramework> {

    DefaultDescription()
    {
    }
};

template<>
struct DefaultDescription<OpenRTB::VideoLinearity>
    : public TaggedEnumDescription<OpenRTB::VideoLinearity> {

    DefaultDescription()
    {
    }
};

template<>
struct DefaultDescription<OpenRTB::VideoBidResponseProtocol>
    : public TaggedEnumDescription<OpenRTB::VideoBidResponseProtocol> {

    DefaultDescription()
    {
    }
};

template<>
struct DefaultDescription<OpenRTB::VideoPlaybackMethod>
    : public TaggedEnumDescription<OpenRTB::VideoPlaybackMethod> {

    DefaultDescription()
    {
    }
};

template<>
struct DefaultDescription<OpenRTB::VideoStartDelay>
    : public TaggedEnumDescription<OpenRTB::VideoStartDelay> {

    DefaultDescription()
    {
    }
};

template<>
struct DefaultDescription<OpenRTB::ConnectionType>
    : public TaggedEnumDescription<OpenRTB::ConnectionType> {

    DefaultDescription()
    {
    }
};

template<>
struct DefaultDescription<OpenRTB::ExpandableDirection>
    : public TaggedEnumDescription<OpenRTB::ExpandableDirection> {

    DefaultDescription()
    {
    }
};

template<>
struct DefaultDescription<OpenRTB::ContentDeliveryMethod>
    : public TaggedEnumDescription<OpenRTB::ContentDeliveryMethod> {

    DefaultDescription()
    {
    }
};

template<>
struct DefaultDescription<OpenRTB::ContentContext>
    : public TaggedEnumDescription<OpenRTB::ContentContext> {

    DefaultDescription()
    {
    }
};

template<>
struct DefaultDescription<OpenRTB::LocationType>
    : public TaggedEnumDescription<OpenRTB::LocationType> {

    DefaultDescription()
    {
    }
};


template<>
struct DefaultDescription<OpenRTB::DeviceType>
    : public TaggedEnumDescription<OpenRTB::DeviceType> {

    DefaultDescription()
    {
    }
};

template<>
struct DefaultDescription<OpenRTB::VastCompanionType>
    : public TaggedEnumDescription<OpenRTB::VastCompanionType> {

    DefaultDescription()
    {
    }
};

template<>
struct DefaultDescription<OpenRTB::MediaRating>
    : public TaggedEnumDescription<OpenRTB::MediaRating> {

    DefaultDescription()
    {
    }
};

template<>
struct DefaultDescription<OpenRTB::FramePosition>
    : public TaggedEnumDescription<OpenRTB::FramePosition> {

    DefaultDescription()
    {
    }
};

template<>
struct DefaultDescription<OpenRTB::SourceRelationship>
    : public TaggedEnumDescription<OpenRTB::SourceRelationship> {

    DefaultDescription()
    {
    }
};

template<>
struct DefaultDescription<OpenRTB::Embeddable>
    : public TaggedEnumDescription<OpenRTB::Embeddable> {

    DefaultDescription()
    {
    }
};

template<>
struct DefaultDescription<OpenRTB::AdPosition>
    : public TaggedEnumDescription<OpenRTB::AdPosition> {

    DefaultDescription()
    {
    }
};
#endif

template<>
struct DefaultDescription<OpenRTB::BidRequest>
    : public StructureDescription<OpenRTB::BidRequest> {
    DefaultDescription();
};

template<>
struct DefaultDescription<OpenRTB::Impression>
    : public StructureDescription<OpenRTB::Impression> {
    DefaultDescription();
};

template<>
struct DefaultDescription<OpenRTB::Banner>
    : public StructureDescription<OpenRTB::Banner> {
    DefaultDescription();
};

template<>
struct DefaultDescription<OpenRTB::Video>
    : public StructureDescription<OpenRTB::Video> {
    DefaultDescription();
};

template<>
struct DefaultDescription<OpenRTB::Content>
    : public StructureDescription<OpenRTB::Content> {
    DefaultDescription();
};

template<>
struct DefaultDescription<OpenRTB::Context>
    : public StructureDescription<OpenRTB::Context> {
    DefaultDescription();
};

template<>
struct DefaultDescription<OpenRTB::Site>
    : public StructureDescription<OpenRTB::Site> {
    DefaultDescription();
};

template<>
struct DefaultDescription<OpenRTB::App>
    : public StructureDescription<OpenRTB::App> {
    DefaultDescription();
};

template<>
struct DefaultDescription<OpenRTB::Device>
    : public StructureDescription<OpenRTB::Device> {
    DefaultDescription();
};

template<>
struct DefaultDescription<OpenRTB::User>
    : public StructureDescription<OpenRTB::User> {
    DefaultDescription();
};

template<>
struct DefaultDescription<OpenRTB::Publisher>
    : public StructureDescription<OpenRTB::Publisher> {
    DefaultDescription();
};

template<>
struct DefaultDescription<OpenRTB::Geo>
    : public StructureDescription<OpenRTB::Geo> {
    DefaultDescription();
};

template<>
struct DefaultDescription<OpenRTB::Data>
    : public StructureDescription<OpenRTB::Data> {
    DefaultDescription();
};

template<>
struct DefaultDescription<OpenRTB::Segment>
    : public StructureDescription<OpenRTB::Segment> {
    DefaultDescription();
};

template<>
struct DefaultDescription<OpenRTB::Bid>
    : public StructureDescription<OpenRTB::Bid> {
    DefaultDescription();
};

template<>
struct DefaultDescription<OpenRTB::SeatBid>
    : public StructureDescription<OpenRTB::SeatBid> {
    DefaultDescription();
};

template<>
struct DefaultDescription<OpenRTB::BidResponse>
    : public StructureDescription<OpenRTB::BidResponse> {
    DefaultDescription();
};

template<>
struct DefaultDescription<OpenRTB::Deal>
    : public StructureDescription<OpenRTB::Deal> {
    DefaultDescription();
};

template<>
struct DefaultDescription<OpenRTB::PMP>
    : public StructureDescription<OpenRTB::PMP> {
    DefaultDescription();
};

template<>
struct DefaultDescription<OpenRTB::Regulations>
    : public StructureDescription<OpenRTB::Regulations> {
    DefaultDescription();
};

} // namespace Datacratic
