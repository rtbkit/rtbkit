/** abort.h                                 -*- C++ -*-
    Rémi Attab, 13 Nov 2012
    Copyright (c) 2012 Datacratic.  All rights reserved.

    Utilities related to the abort() function.

    These functions are meant to be used as debugging helpers so that the
    program can be stopped as soon as an error is detected. This is mainly
    useful if we can't use "catch throw" in gdb because it would yield too many
    false positives. Instead we just sprinkle several calls to do_abort() and
    let gdb break on SIGABRT.

*/

#ifndef __jml__utils__abort_h__
#define __jml__utils__abort_h__

namespace ML {

/** Calls abort() if one of the following criterias are met:

    - The environment variable JML_ABORT is set.
    - The macro JML_ABORT is defined.
    - set_abort_state(true) is called.

    Note that the value passed to set_abort_state() will override all other
    settings. Also note that the environmnent variable is only read once at
    startup in a static constructor.

 */
void do_abort();

/** If false, do_abort() will not call abort(). If true, the oposite happens. */
bool get_abort_state();

/** Overides the current behaviour of the do_abort() function. */
void set_abort_state(bool b);


} // ML

#endif // __jml__utils__abort_h__
