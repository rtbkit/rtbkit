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
struct DefaultDescription<OpenRTB::TaggedBool>
    : public ValueDescriptionT<OpenRTB::TaggedBool> {
  
    virtual void parseJsonTyped(OpenRTB::TaggedBool * val,
                                JsonParsingContext & context) const
    {
        if (context.isBool())
            val->val = context.expectBool();
        else val->val = context.expectInt();
    }

    virtual void printJsonTyped(const OpenRTB::TaggedBool * val,
                                JsonPrintingContext & context) const
    {
        context.writeInt(val->val);
    }

    virtual bool isDefaultTyped(const OpenRTB::TaggedBool * val)
        const
    {
        return val->val == -1;
    }
};

template<int defValue>
struct DefaultDescription<OpenRTB::TaggedBoolDef<defValue> >
    : public ValueDescriptionT<OpenRTB::TaggedBoolDef<defValue> > {
  
    virtual void parseJsonTyped(OpenRTB::TaggedBoolDef<defValue> * val,
                                JsonParsingContext & context) const
    {
        if (context.isBool())
            val->val = context.expectBool();
        else val->val = context.expectInt();
    }

    virtual void printJsonTyped(const OpenRTB::TaggedBoolDef<defValue> * val,
                                JsonPrintingContext & context) const
    {
        context.writeInt(val->val);
    }

    virtual bool isDefaultTyped(const OpenRTB::TaggedBoolDef<defValue> * val)
        const
    {
        return val->val == defValue;
    }
};

template<>
struct DefaultDescription<OpenRTB::TaggedInt>
    : public ValueDescriptionT<OpenRTB::TaggedInt> {

    virtual void parseJsonTyped(OpenRTB::TaggedInt * val,
                                JsonParsingContext & context) const
    {
        if (context.isString()) {
            std::string s = context.expectStringAscii();
            val->val = boost::lexical_cast<int>(s);
        }
        else val->val = context.expectInt();
    }

    virtual void printJsonTyped(const OpenRTB::TaggedInt * val,
                                JsonPrintingContext & context) const
    {
        context.writeInt(val->val);
    }

    virtual bool isDefaultTyped(const OpenRTB::TaggedInt * val)
        const
    {
        return val->val == -1;
    }
};

template<int defValue>
struct DefaultDescription<OpenRTB::TaggedIntDef<defValue> >
    : public ValueDescriptionT<OpenRTB::TaggedIntDef<defValue> > {

    virtual void parseJsonTyped(OpenRTB::TaggedIntDef<defValue> * val,
                                JsonParsingContext & context) const
    {
        if (context.isString()) {
            std::string s = context.expectStringAscii();
            val->val = boost::lexical_cast<int>(s);
        }
        else val->val = context.expectInt();
    }

    virtual void printJsonTyped(const OpenRTB::TaggedIntDef<defValue> * val,
                                JsonPrintingContext & context) const
    {
        context.writeInt(val->val);
    }

    virtual bool isDefaultTyped(const OpenRTB::TaggedIntDef<defValue> * val)
        const
    {
        return val->val == defValue;
    }
};

template<>
struct DefaultDescription<OpenRTB::TaggedFloat>
    : public ValueDescriptionT<OpenRTB::TaggedFloat> {

    virtual void parseJsonTyped(OpenRTB::TaggedFloat * val,
                                JsonParsingContext & context) const
    {
        val->val = context.expectFloat();
    }

    virtual void printJsonTyped(const OpenRTB::TaggedFloat * val,
                                JsonPrintingContext & context) const
    {
        context.writeDouble(val->val);
    }

    virtual bool isDefaultTyped(const OpenRTB::TaggedFloat * val) const
    {
        return isnan(val->val);
    }
};

template<int num, int den>
struct DefaultDescription<OpenRTB::TaggedFloatDef<num, den> >
    : public ValueDescriptionT<OpenRTB::TaggedFloatDef<num, den> > {

    virtual void parseJsonTyped(OpenRTB::TaggedFloatDef<num, den> * val,
                                JsonParsingContext & context) const
    {
        val->val = context.expectFloat();
    }

    virtual void printJsonTyped(const OpenRTB::TaggedFloatDef<num, den> * val,
                                JsonPrintingContext & context) const
    {
        context.writeFloat(val->val);
    }

    virtual bool isDefaultTyped(const OpenRTB::TaggedFloatDef<num, den> * val) const
    {
        return val->val == (float)num / den;
    }
};

template<class Enum>
struct TaggedEnumDescription
    : public ValueDescriptionT<Enum> {

    virtual void parseJsonTyped(Enum * val,
                                JsonParsingContext & context) const
    {
        int index = context.expectInt();
        val->val = index;
    }

    virtual void printJsonTyped(const Enum * val,
                                JsonPrintingContext & context) const
    {
        context.writeInt(val->val);
    }

    virtual bool isDefaultTyped(const Enum * val) const
    {
        return val->val == -1;
    }
};

/// Parser for banner w or h.  This can either be:
/// single format: "w": 123
/// multiple formats: "w": [ 123, 456 ]
struct FormatListDescription
    : public ValueDescriptionT<OpenRTB::List<int> >,
      public ListDescriptionBase<int> {

    virtual void parseJsonTyped(OpenRTB::List<int> * val,
                                JsonParsingContext & context) const
    {
        if (context.isArray()) {
            auto onElement = [&] ()
                {
                    val->push_back(context.expectInt());
                };
            context.forEachElement(onElement);
        }
        else {
            val->push_back(context.expectInt());
        }
    }

    virtual void printJsonTyped(const OpenRTB::List<int> * val,
                                JsonPrintingContext & context) const
    {
        if (val->size() == 1) {
            this->inner->printJsonTyped(&(*val)[0], context);
        }
        else
            printJsonTypedList(val, context);
    }

    virtual bool isDefaultTyped(const OpenRTB::List<int> * val) const
    {
        return val->empty();
    }

};

struct CommaSeparatedListDescription
    : public ValueDescriptionT<std::string> {

    virtual void parseJsonTyped(std::string * val,
                                JsonParsingContext & context) const
    {
        if (context.isArray()) {
            std::string res;
            auto onElement = [&] ()
                {
                    std::string s = context.expectStringAscii();
                    if (!res.empty())
                        res += ", ";
                    res += s;
                        
                };
            context.forEachElement(onElement);
            *val = res;
        }
        else {
            *val = context.expectStringAscii();
        }
    }

    virtual void printJsonTyped(const std::string * val,
                                JsonPrintingContext & context) const
    {
        context.writeString(*val);
    }

    virtual bool isDefaultTyped(const std::string * val) const
    {
        return val->empty();
    }

};

template<typename T>
struct DefaultDescription<OpenRTB::Optional<T> >
    : public ValueDescriptionT<OpenRTB::Optional<T> > {

    DefaultDescription(ValueDescriptionT<T> * inner
                       = getDefaultDescription((T *)0))
        : inner(inner)
    {
    }

    std::unique_ptr<ValueDescriptionT<T> > inner;

    virtual void parseJsonTyped(OpenRTB::Optional<T> * val,
                                JsonParsingContext & context) const
    {
        val->reset(new T());
        inner->parseJsonTyped(val->get(), context);
    }

    virtual void printJsonTyped(const OpenRTB::Optional<T> * val,
                                JsonPrintingContext & context) const
    {
        if (!val->get())
            context.skip();
        else inner->printJsonTyped(val->get(), context);
    }

    virtual bool isDefaultTyped(const OpenRTB::Optional<T> * val) const
    {
        return !val->get();
    }
};

template<typename T>
struct DefaultDescription<OpenRTB::List<T> >
    : public ValueDescriptionT<OpenRTB::List<T> >,
      public ListDescriptionBase<T> {

    virtual void parseJsonTyped(OpenRTB::List<T> * val,
                                JsonParsingContext & context) const
    {
        this->parseJsonTypedList(val, context);
    }

    virtual void printJsonTyped(const OpenRTB::List<T> * val,
                                JsonPrintingContext & context) const
    {
        this->printJsonTypedList(val, context);
    }

    virtual bool isDefaultTyped(const OpenRTB::List<T> * val) const
    {
        return val->empty();
    }
};

template<>
struct DefaultDescription<OpenRTB::ContentCategory>
    : public ValueDescriptionT<OpenRTB::ContentCategory> {

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
    : public ValueDescriptionT<OpenRTB::MimeType> {

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
struct DefaultDescription<OpenRTB::AdPosition>
    : public TaggedEnumDescription<OpenRTB::AdPosition> {

    DefaultDescription()
    {
    }
};

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




} // namespace Datacratic
