/* stats_events.h                                                  -*- C++ -*-
   Jeremy Barnes, 1 March 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Different types of stats events.
*/

#ifndef __logger__stats_events_h__
#define __logger__stats_events_h__

namespace Datacratic {

enum StatEventType {
    ET_HIT,          ///< Represents an extra count on a counter
    ET_COUNT,        ///< Represents an extra value accumulated
    ET_STABLE_LEVEL, ///< Represents the current level of a stable something
    ET_LEVEL,        ///< Represents the current level of something
    ET_OUTCOME       ///< Represents the outcome of an experiment
};

static constexpr std::initializer_list<int> DefaultOutcomePercentiles =
    { 90, 95, 98 };

} // namespace Datacratic



#endif /* __logger__stats_events_h__ */
