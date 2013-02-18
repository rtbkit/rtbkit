/* stats_events.h                                                  -*- C++ -*-
   Jeremy Barnes, 1 March 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Different types of stats events.
*/

#ifndef __logger__stats_events_h__
#define __logger__stats_events_h__

namespace Datacratic {

enum EventType {
    ET_COUNT,    ///< Represents an extra count on a counter
    ET_ACCUM,    ///< Represents an extra value accumulated
    ET_LEVEL,    ///< Represents the current level of something
    ET_OUTCOME   ///< Represents the outcome of an experiment
};


} // namespace Datacratic



#endif /* __logger__stats_events_h__ */
