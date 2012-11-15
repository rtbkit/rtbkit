/** abort.cc                                 -*- C++ -*-
    Rémi Attab, 13 Nov 2012
    Copyright (c) 2012 Datacratic.  All rights reserved.

    Utilities related to the abort() function

*/

#include "abort.h"

#include <cstdlib>
#include <iostream>

using namespace std;


namespace ML {

namespace {


/******************************************************************************/
/* COMPILE SETTING                                                            */
/******************************************************************************/

#ifndef JML_ABORT
#   define JML_ABORT 0
#else
#   undef JML_ABORT
#   define JML_ABORT 1
#endif

enum { COMPILE_STATE = JML_ABORT };


/******************************************************************************/
/* ABORT STATE                                                                */
/******************************************************************************/

struct AbortState {

    AbortState() :
        state(COMPILE_STATE)
    {
        state = state || getenv("JML_ABORT") != NULL;
    }

    bool state;
} staticState;

}; // namespace anonymous


/******************************************************************************/
/* INTERFACE                                                                  */
/******************************************************************************/

void do_abort()
{
    if (staticState.state) abort();
}

bool get_abort_state()
{
    return staticState.state;
}

void set_abort_state(bool b)
{
    staticState.state = b;
}



} // namepsace ML
