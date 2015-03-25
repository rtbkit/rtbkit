/** priority.h                                 -*- C++ -*-
    Rémi Attab, 09 Aug 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Priority of the various filters.

*/

#pragma once

namespace RTBKIT {


/******************************************************************************/
/* FILTER PRIORITY                                                            */
/******************************************************************************/

struct Priority
{
    static constexpr unsigned ExchangeName         = 0x0200;

    static constexpr unsigned Location             = 0x1000;
    static constexpr unsigned Language             = 0x1100;
    static constexpr unsigned Host                 = 0x1200;
    static constexpr unsigned Url                  = 0x1300;

    static constexpr unsigned CreativeFormat       = 0x2000;
    static constexpr unsigned CreativeLocation     = 0x2100;
    static constexpr unsigned CreativeExchangeName = 0x2200;
    static constexpr unsigned CreativeLanguage     = 0x2300;
    static constexpr unsigned CreativePMP          = 0x2400;

    static constexpr unsigned Segments             = 0x3000;
    static constexpr unsigned HourOfWeek           = 0x3100;
    static constexpr unsigned FoldPosition         = 0x3200;
    static constexpr unsigned RequiredIds          = 0x3300;
    static constexpr unsigned UserPartition        = 0x3400;

    static constexpr unsigned CreativeSegments     = 0x3500;

    static constexpr unsigned ExchangePre          = 0xF000;

    // Really slow so delay as much as possible.
    static constexpr unsigned CreativeExchange     = 0xF100;

    static constexpr unsigned LatLong              = 0xF200;

    static constexpr unsigned ExchangePost         = 0xFF00;
};


} // namespace RTBKIT
