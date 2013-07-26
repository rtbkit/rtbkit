/* openrtb_parsing.h                                               -*- C++ -*-
   Jeremy Barnes, 22 February 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Code to parse OpenRTB bid requests.
*/

#pragma once

#include "openrtb.h"
#include "soa/types/value_description.h"
#include "common/descriptions/autocast.h"
#include "common/cpp11.h"

namespace Datacratic
{
	/// Parser for banner w or h.  This can either be:
	/// single format: "w": 123
	/// multiple formats: "w": [ 123, 456 ]
	struct FormatListDescription: public ValueDescriptionI<std::vector<int>>, public ListDescriptionBase<int>
	{
		virtual void parseJsonTyped(std::vector<int>* val, JsonParsingContext& context) const override
		{
			if (context.isArray())
				context.forEachElement([&]()
				{
					val->push_back(context.expectInt());
				});
			else
				val->push_back(context.expectInt());
		}

		virtual void printJsonTyped(const std::vector<int>* val, JsonPrintingContext& context) const override
		{
			if (val->size() == 1)
				this->inner->printJsonTyped(&(*val)[0], context);
			else
				printJsonTypedList(val, context);
		}

		virtual bool isDefaultTyped(const std::vector<int>* val) const override
		{
			return val->empty();
		}
	};

	struct CommaSeparatedListDescription: public ValueDescriptionI<std::string, ValueKind::STRING>
	{
		virtual void parseJsonTyped(std::string* val, JsonParsingContext& context) const override
		{
			if (context.isArray()) {
				std::string res;
				context.forEachElement([&]()
				{
					std::string s = context.expectString();
					if (!res.empty())
						res += ", ";
					res += s;
				});
				*val = res;
			}
			else {
				*val = context.expectStringAscii();
			}
		}

		virtual void printJsonTyped(const std::string* val, JsonPrintingContext& context) const override
		{
			context.writeString(*val);
		}

		virtual bool isDefaultTyped(const std::string* val) const override
		{
			return val->empty();
		}
	};

	inline OpenRTB::Currency toCurrency(const std::string& str)
	{
		return OpenRTB::MakeCurrency(str.c_str());
	}

	inline std::string toString(OpenRTB::Currency cur)
	{
		union {
			char s[5];
			uint32_t c;
		};

		c = uint32_t(cur);
		s[4] = '\0';

		return std::string(s);
	}

	template<> struct DefaultDescription<OpenRTB::Currency> : public StringEnumDescription<OpenRTB::Currency, toCurrency, toString> {};
	template<> struct DefaultDescription<OpenRTB::BannerAdType> : public IntegerEnumDescription<OpenRTB::BannerAdType> {};
	template<> struct DefaultDescription<OpenRTB::CreativeAttribute> : public IntegerEnumDescription<OpenRTB::CreativeAttribute> {};
	template<> struct DefaultDescription<OpenRTB::ApiFramework> : public IntegerEnumDescription<OpenRTB::ApiFramework> {};
	template<> struct DefaultDescription<OpenRTB::AdPosition> : public IntegerEnumDescription<OpenRTB::AdPosition> {};
	template<> struct DefaultDescription<OpenRTB::VideoLinearity> : public IntegerEnumDescription<OpenRTB::VideoLinearity> {};
	template<> struct DefaultDescription<OpenRTB::VideoBidResponseProtocol> : public IntegerEnumDescription<OpenRTB::VideoBidResponseProtocol> {};
	template<> struct DefaultDescription<OpenRTB::VideoPlaybackMethod> : public IntegerEnumDescription<OpenRTB::VideoPlaybackMethod> {};
	template<> struct DefaultDescription<OpenRTB::VideoStartDelay> : public IntegerEnumDescription<OpenRTB::VideoStartDelay> {};
	template<> struct DefaultDescription<OpenRTB::ConnectionType> : public IntegerEnumDescription<OpenRTB::ConnectionType> {};
	template<> struct DefaultDescription<OpenRTB::ExpandableDirection> : public IntegerEnumDescription<OpenRTB::ExpandableDirection> {};
	template<> struct DefaultDescription<OpenRTB::ContentDeliveryMethod> : public IntegerEnumDescription<OpenRTB::ContentDeliveryMethod> {};
	template<> struct DefaultDescription<OpenRTB::ContentContext> : public IntegerEnumDescription<OpenRTB::ContentContext> {};
	template<> struct DefaultDescription<OpenRTB::VideoQuality> : public IntegerEnumDescription<OpenRTB::VideoQuality> {};
	template<> struct DefaultDescription<OpenRTB::LocationType> : public IntegerEnumDescription<OpenRTB::LocationType> {};
	template<> struct DefaultDescription<OpenRTB::DeviceType> : public IntegerEnumDescription<OpenRTB::DeviceType> {};
	template<> struct DefaultDescription<OpenRTB::VastCompanionType> : public IntegerEnumDescription<OpenRTB::VastCompanionType> {};
	template<> struct DefaultDescription<OpenRTB::MediaRating> : public IntegerEnumDescription<OpenRTB::MediaRating> {};
	template<> struct DefaultDescription<OpenRTB::AuctionType> : public IntegerEnumDescription<OpenRTB::AuctionType> {};
	template<> struct DefaultDescription<OpenRTB::Content::SourceRelationship> : public IntegerEnumDescription<OpenRTB::Content::SourceRelationship> {};

	template<> struct DefaultDescription<OpenRTB::BidRequest> : public AutoCastingStructureDescription<OpenRTB::BidRequest> {DefaultDescription();};
	template<> struct DefaultDescription<OpenRTB::Impression> : public AutoCastingStructureDescription<OpenRTB::Impression> {DefaultDescription();};
	template<> struct DefaultDescription<OpenRTB::Banner> : public AutoCastingStructureDescription<OpenRTB::Banner> {DefaultDescription();};
	template<> struct DefaultDescription<OpenRTB::Video> : public AutoCastingStructureDescription<OpenRTB::Video> {DefaultDescription();};
	template<> struct DefaultDescription<OpenRTB::Content> : public AutoCastingStructureDescription<OpenRTB::Content> {DefaultDescription();};
	template<> struct DefaultDescription<OpenRTB::Context> : public AutoCastingStructureDescription<OpenRTB::Context> {DefaultDescription();};
	template<> struct DefaultDescription<OpenRTB::Site> : public AutoCastingStructureDescription<OpenRTB::Site> {DefaultDescription();};
	template<> struct DefaultDescription<OpenRTB::App> : public AutoCastingStructureDescription<OpenRTB::App> {DefaultDescription();};
	template<> struct DefaultDescription<OpenRTB::Device> : public AutoCastingStructureDescription<OpenRTB::Device> {DefaultDescription();};
	template<> struct DefaultDescription<OpenRTB::User> : public AutoCastingStructureDescription<OpenRTB::User> {DefaultDescription();};
	template<> struct DefaultDescription<OpenRTB::Publisher> : public AutoCastingStructureDescription<OpenRTB::Publisher> {DefaultDescription();};
	template<> struct DefaultDescription<OpenRTB::Geo> : public AutoCastingStructureDescription<OpenRTB::Geo> {DefaultDescription();};
	template<> struct DefaultDescription<OpenRTB::Data> : public AutoCastingStructureDescription<OpenRTB::Data> {DefaultDescription();};
	template<> struct DefaultDescription<OpenRTB::Segment> : public AutoCastingStructureDescription<OpenRTB::Segment> {DefaultDescription();};
	template<> struct DefaultDescription<OpenRTB::Bid> : public AutoCastingStructureDescription<OpenRTB::Bid> {DefaultDescription();};
	template<> struct DefaultDescription<OpenRTB::SeatBid> : public AutoCastingStructureDescription<OpenRTB::SeatBid> {DefaultDescription();};
	template<> struct DefaultDescription<OpenRTB::BidResponse> : public AutoCastingStructureDescription<OpenRTB::BidResponse> {DefaultDescription();};

} // namespace Datacratic
