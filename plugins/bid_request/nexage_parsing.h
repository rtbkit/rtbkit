/* nexage_parsing.h                                               -*- C++ -*-
   Syed Murtaza Hussain Kazmi, 11 September 2013
   Copyright (c) 2013 Virtual Force.  All rights reserved.

   Code to parse Nexage bid requests.
*/

#pragma once

#include <string>
#include "nexage.h"
#include "openrtb/openrtb_parsing.h"

using std::string;


namespace Datacratic {


template<>
struct DefaultDescription<Nexage::NexageLocationType> : public TaggedEnumDescription<Nexage::NexageLocationType> {

    DefaultDescription()
    {
    }

    void parseJsonTyped(Nexage::NexageLocationType * val,
                   JsonParsingContext & context) const
    {
		
		string NexageType;

		if (context.isString())
		{
			NexageType = context.expectStringAscii();
		}
		else
		{
			
			 NexageType = boost::lexical_cast<string>(context.expectInt());	
		}

		//	std::cerr << std::endl << std::endl << "Type=====>>>>>>>" << NexageType << std::endl<< std::endl<< std::endl;

        if (NexageType == "-1") {
          val->val = Nexage::NexageLocationType::UNSPECIFIED;
        } 
        else if (NexageType == "1") {
          val->val = Nexage::NexageLocationType::GPS;
        } 
        else if (NexageType == "2") {
          val->val = Nexage::NexageLocationType::IP_ADDRESS;
        }
        else if (NexageType == "3") {
		 	
          val->val = Nexage::NexageLocationType::USER;
		 
        }
    }

};

	template<>
	struct DefaultDescription<Nexage::BidRequest> : public StructureDescription<Nexage::BidRequest> {
			DefaultDescription();
	};

	template<>
	struct DefaultDescription<Nexage::NexageGeo>
	: public StructureDescription<Nexage::NexageGeo> {
	DefaultDescription();
	};

	template<>
	struct DefaultDescription<Nexage::NexageUser>
	: public StructureDescription<Nexage::NexageUser> {
	DefaultDescription();
	};

} // namespace Datacratic
