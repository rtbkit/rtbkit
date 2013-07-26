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

OpenRTB::AdPosition convertAdPosition(std::string pos) {
    if (pos == "above") {
        return OpenRTB::AdPosition::ABOVE;
    } 
    else if (pos == "below") {
        return OpenRTB::AdPosition::BELOW;
    }
    else { // AN only supports the above two AdPos types.
           // ORTB supports others but AN does not.
        return OpenRTB::AdPosition::UNKNOWN;
    }
}

} // namespace RTBKIT

