/* exc_assert.h                                                    -*- C++ -*-
   Jeremy Barnes, 15 July 2010
   Copyright (c) 2010 Recoset.  All rights reserved.
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   Simple functionality to include asserts that throw exceptions rather than
   abort the program.
*/

#ifndef __jml__utils__exc_assert_h__
#define __jml__utils__exc_assert_h__

#include "jml/arch/exception.h"

namespace ML {

struct Assertion_Failure: public Exception {
    Assertion_Failure(const std::string & msg);
    Assertion_Failure(const char * msg, ...);
    Assertion_Failure(const char * assertion,
                      const char * function,
                      const char * file,
                      int line);
};

} // namespace ML

// Assert that throws an exception if it doesn't hold instead of aborting
#define ExcAssert(condition)                                        \
    do {                                                            \
        if (!(condition))                                           \
            throw ML::Assertion_Failure(#condition, __PRETTY_FUNCTION__, \
                                        __FILE__, __LINE__);            \
    } while (0)





#endif /* __jml__utils__exc_assert_h__ */
