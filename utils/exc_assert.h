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
#include "jml/arch/format.h"
#include <boost/lexical_cast.hpp>

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

/// Assert that the two values are equal.  They must not have any side
/// effects as they may be evaluated more than once.
#define ExcAssertOp(op, value1, value2)                                 \
    do {                                                                \
        if (!((value1) op (value2))) {                                  \
            std::string v1 = boost::lexical_cast<std::string>(value1);  \
            std::string v2 = boost::lexical_cast<std::string>(value2);  \
            std::string msg = ML::format("!(%s " #op " %s) [!(%s " #op " %s)]", \
                                         #value1, #value2,              \
                                         v1.c_str(), v2.c_str());       \
            throw ML::Assertion_Failure(msg.c_str(), __PRETTY_FUNCTION__, \
                                        __FILE__, __LINE__);            \
        } \
    } while (0)

/// Assert that the two values are equal.  They must not have any side
/// effects as they may be evaluated more than once.
#define ExcAssertEqual(value1, value2) \
    ExcAssertOp(==, value1, value2)

/// Assert that the two values are equal.  They must not have any side
/// effects as they may be evaluated more than once.
#define ExcAssertNotEqual(value1, value2) \
    ExcAssertOp(!=, value1, value2)

#define ExcAssertLessEqual(value1, value2) \
    ExcAssertOp(<=, value1, value2)

#define ExcAssertLess(value1, value2) \
    ExcAssertOp(<, value1, value2)

#define ExcAssertGreaterEqual(value1, value2) \
    ExcAssertOp(>=, value1, value2)

#define ExcAssertGreater(value1, value2) \
    ExcAssertOp(>, value1, value2)



#endif /* __jml__utils__exc_assert_h__ */
