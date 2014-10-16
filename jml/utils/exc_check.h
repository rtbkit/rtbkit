/* exc_check.h                                                    -*- C++ -*-
   Rémi Attab, 24 Febuary 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Quick and easy way to throw an exception on a failed condition.

   Note that when a check fails, the macro will issue a call to ML::do_abort().
   This makes the macros more debugger friendly in certain situations.

*/


#ifndef __jml__utils__exc_check_h__
#define __jml__utils__exc_check_h__

#include "abort.h"
#include "jml/arch/exception.h"
#include "jml/arch/format.h"
#include <boost/lexical_cast.hpp>
#include <errno.h>

namespace ML {

struct Check_Failure: public Exception {
    Check_Failure(const std::string & msg);
    Check_Failure(const char * msg, ...);
    Check_Failure(const char * assertion,
                  const char * function,
                  const char * file,
                  int line);
};

} // namespace ML

/// Throws a formatted exception if the condition is false.
#define ExcCheckImpl(condition, message, exc_type)                      \
    do {                                                                \
        if (!(condition)) {                                             \
            std::string msg__ =                                         \
                ML::format("%s: %s", message, #condition);              \
            ML::do_abort();                                             \
            throw exc_type(msg__.c_str(), __PRETTY_FUNCTION__,          \
                    __FILE__, __LINE__);                                \
        }                                                               \
    } while (0)

/// Check that the two values meet the operand.  
/// They must not have any side effects as they may be evaluated more than once.
#define ExcCheckOpImpl(op, value1, value2, message, exc_type)           \
    do {                                                                \
        if (!((value1) op (value2))) {                                  \
            std::string v1__ = boost::lexical_cast<std::string>(value1);\
            std::string v2__ = boost::lexical_cast<std::string>(value2);\
            std::string msg__ = ML::format(                             \
                    "%s: !(%s " #op " %s) [!(%s " #op " %s)]",          \
                    message, #value1, #value2, v1__.c_str(),            \
                    v2__.c_str());                                      \
            ML::do_abort();                                             \
            throw exc_type(msg__.c_str(), __PRETTY_FUNCTION__,          \
                                        __FILE__, __LINE__);            \
        }                                                               \
    } while (0)

/// Throws a formatted exception if the condition is false.
#define ExcCheckErrnoImpl(condition, message, exc_type)                 \
    do {                                                                \
        if (!(condition)) {                                             \
            std::string msg__ = ML::format("%s: %s(%d)",                \
                    message, strerror(errno), errno);                   \
            ML::do_abort();                                             \
            throw exc_type(msg__.c_str(), __PRETTY_FUNCTION__,          \
                    __FILE__, __LINE__);                                \
        }                                                               \
    } while (0)


/// Simple forwarders with the right exception type.
#define ExcCheck(condition, message)                    \
    ExcCheckImpl(condition, message, ML::Check_Failure)

#define ExcCheckOp(op, value1, value2, message)                         \
    ExcCheckOpImpl(op, value1, value2, message, ML::Check_Failure)

#define ExcCheckErrno(condition, message)       \
    ExcCheckErrnoImpl(condition, message, ML::Check_Failure)


/// see ExcCheckOpImpl for more details
#define ExcCheckEqual(value1, value2, message)  \
    ExcCheckOp(==, value1, value2, message)

#define ExcCheckNotEqual(value1, value2, message)       \
    ExcCheckOp(!=, value1, value2, message)

#define ExcCheckLessEqual(value1, value2, message)      \
    ExcCheckOp(<=, value1, value2, message)

#define ExcCheckLess(value1, value2, message)   \
    ExcCheckOp(<, value1, value2, message)

#define ExcCheckGreaterEqual(value1, value2, message)   \
    ExcCheckOp(>=, value1, value2, message)

#define ExcCheckGreater(value1, value2, message)        \
    ExcCheckOp(>, value1, value2, message)



#endif /* __jml__utils__exc_check_h__ */
