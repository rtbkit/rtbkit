/* exc_assert.h                                                    -*- C++ -*-
   Jeremy Barnes, 15 July 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   Simple functionality to include asserts that throw exceptions rather than
   abort the program.
*/

#ifndef __jml__utils__exc_assert_h__
#define __jml__utils__exc_assert_h__

#include "jml/arch/exception.h"
#include "jml/utils/exc_check.h"

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

/// Simple forwarders with the right exception type.
/// These will eventually be disable-able using a compiler switch.
#define ExcAssert(condition)                    \
    ExcCheckImpl(condition, "Assert failure", ML::Assertion_Failure)

#define ExcAssertOp(op, value1, value2)         \
    ExcCheckOpImpl(op, value1, value2, "Assert failure", ML::Assertion_Failure)

#define ExcAssertErrno(condition)               \
    ExcCheckErrnoImpl(condition, "Assert failure", ML::Assertion_Failure)

/// see ExcCheckOpImpl for more details
#define ExcAssertEqual(value1, value2)          \
    ExcAssertOp(==, value1, value2)

#define ExcAssertNotEqual(value1, value2)       \
    ExcAssertOp(!=, value1, value2)

#define ExcAssertLessEqual(value1, value2)      \
    ExcAssertOp(<=, value1, value2)

#define ExcAssertLess(value1, value2)           \
    ExcAssertOp(<, value1, value2)

#define ExcAssertGreaterEqual(value1, value2)   \
    ExcAssertOp(>=, value1, value2)

#define ExcAssertGreater(value1, value2)        \
    ExcAssertOp(>, value1, value2)



#endif /* __jml__utils__exc_assert_h__ */
