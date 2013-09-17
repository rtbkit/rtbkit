/** creative_filters.cc                                 -*- C++ -*-
    Rémi Attab, 09 Aug 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Registry and implementation of the creative filters.

*/

#include "creative_filters.h"

using namespace std;
using namespace ML;

namespace RTBKIT {

/******************************************************************************/
/* INIT FILTERS                                                               */
/******************************************************************************/

namespace {

struct InitFilters
{
    InitFilters()
    {
        RTBKIT::FilterRegistry::registerFilter<RTBKIT::CreativeFormatFilter>();
        RTBKIT::FilterRegistry::registerFilter<RTBKIT::CreativeLanguageFilter>();
        RTBKIT::FilterRegistry::registerFilter<RTBKIT::CreativeLocationFilter>();

        RTBKIT::FilterRegistry::registerFilter<RTBKIT::CreativeExchangeNameFilter>();
        RTBKIT::FilterRegistry::registerFilter<RTBKIT::CreativeExchangeFilter>();
    }

} initFilters;

} // namespace anonymous

} // namepsace RTBKIT
