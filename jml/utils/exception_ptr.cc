/* exception_ptr.cc
   Wolfgang Sourdeau, July 2015
   Copyright (c) 2015 Datacratic.  All rights reserved.

*/

#include "exception_ptr.h"


using namespace std;
using namespace ML;


/****************************************************************************/
/* EXCEPTIONPTR HANDLER                                                     */
/****************************************************************************/

bool
ExceptionPtrHandler::
hasException()
{
    std::unique_lock<mutex> guard(excLock_);
    return bool(excPtr_);
}

void
ExceptionPtrHandler::
takeException(std::exception_ptr newPtr)
{
    std::unique_lock<mutex> guard(excLock_);
    excPtr_ = newPtr;
}

void
ExceptionPtrHandler::
takeCurrentException()
{
    takeException(std::current_exception());
}

void
ExceptionPtrHandler::
rethrowIfSet()
{
    std::unique_lock<mutex> guard(excLock_);
    if (excPtr_) {
        std::exception_ptr ptr = excPtr_;
        excPtr_ = nullptr;
        std::rethrow_exception(ptr);
    }
}
