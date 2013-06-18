/** appnexus_openrtb_mapping.h                                                      -*- C++ -*-
    Mark Weiss, 8 May 2013
    Copyright (c) 2013 Datacratic Inc.  All rights reserved.

    This file is part of RTBkit.

    Mapping functions mapping the OpenRTB bid request
    structs to the AppNexus bid request structs
*/

#pragma once

#include "plugins/bid_request/appnexus.h"
#include "openrtb/openrtb.h"


namespace RTBKIT {

OpenRTB::AdPosition convertAdPosition(AppNexus::AdPosition pos) {
    OpenRTB::AdPosition ret;

    if (pos.value() == AppNexus::AdPosition::UNKNOWN) {
        ret.val = OpenRTB::AdPosition::UNKNOWN;
    } 
    else if (pos.value() == AppNexus::AdPosition::ABOVE) {
        ret.val = OpenRTB::AdPosition::ABOVE;
    } 
    else if (pos.value() == AppNexus::AdPosition::BELOW) {
        ret.val = OpenRTB::AdPosition::BELOW;
    }
    else { // AN only supports the above three AdPos types.
           // ORTB supports others but AN does not.
        ret.val = OpenRTB::AdPosition::UNSPECIFIED;
    }

    return ret;
}

} // namespace RTBKIT

