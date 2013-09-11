/* value_description.h                                             -*- C++ -*-
   Wolfgang Sourdeau, September 2013
   Copyright (C) 2013 Datacratic.  All rights reserved.

*/

#pragma once

#include "soa/types/value_description.h"

#include "standard_events.h"


namespace Datacratic {

template<>
struct DefaultDescription<RTBKIT::StandardWin>
    : public StructureDescription<RTBKIT::StandardWin> {
    DefaultDescription();
};

template<>
struct DefaultDescription<RTBKIT::StandardExternalWin>
    : public StructureDescription<RTBKIT::StandardExternalWin> {
    DefaultDescription();
};

template<>
struct DefaultDescription<RTBKIT::StandardClick>
    : public StructureDescription<RTBKIT::StandardClick> {
    DefaultDescription();
};

template<>
struct DefaultDescription<RTBKIT::StandardConversion>
    : public StructureDescription<RTBKIT::StandardConversion> {
    DefaultDescription();
};

} // namespace Datacratic
