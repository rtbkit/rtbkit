/** future_fix.h                                 -*- C++ -*-
    Rémi Attab, 26 Jul 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Fix for the gcc inconcistencies for std::future

*/

#pragma once

#include <future>

namespace Datacratic {


#define GCC_VERSION (__GNUC__ * 10000       \
                     + __GNUC_MINOR__ * 100 \
                     + __GNUC_PATCHLEVEL__)


/******************************************************************************/
/* FUTURE WAIT FOR                                                            */
/******************************************************************************/

template<typename FutT, typename DurT, typename DurScale>
bool
wait_for(std::future<FutT>& f, const std::chrono::duration<DurT, DurScale>& d)
{

#if GCC_VERSION >= 40700
    return f.wait_for(d) == std::future_status::ready;

#else
    return f.wait_for(d);

#endif
}


#undef GCC_VERSION

} // namespace Datacratic
