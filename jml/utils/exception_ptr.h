/* exception_ptr.h                                                 -*- C++ -*-
   Wolfgang Sourdeau, July 2015
   Copyright (c) 2015 Datacratic.  All rights reserved.

*/

#pragma once

#include <exception>
#include <mutex>


namespace ML {

/****************************************************************************/
/* EXCEPTIONPTR HANDLER                                                     */
/****************************************************************************/

/* This class provides thread-safe handling of exception-ptr. */
struct ExceptionPtrHandler {
    bool hasException();
    void takeException(std::exception_ptr newPtr);
    void takeCurrentException();
    void rethrowIfSet();
    void clear()
    { takeException(nullptr); }

private:
    std::mutex excLock_;
    std::exception_ptr excPtr_;
};

} // namespace Datacratic
