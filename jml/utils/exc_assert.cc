/* exc_assert.cc
   Jeremy Barnes, 15 July 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.
   Copyright (c) 2010 Datacratic Inc.  All rights reserved.

*/

#include "exc_assert.h"
#include "jml/arch/format.h"

namespace ML {

Assertion_Failure::
Assertion_Failure(const std::string & msg)
    : Exception(msg)
{
}

Assertion_Failure::
Assertion_Failure(const char * msg, ...)
    : Exception(msg)
{
}

Assertion_Failure::
Assertion_Failure(const char * assertion,
                  const char * function,
                  const char * file,
                  int line)
    : Exception(format("assertion failure: %s at %s:%d in %s",
                    assertion, file, line, function))
{
}

} // namespace ML
