/* exception_internals.h                                           -*- C++ -*-
   Jeremy Barnes, 18 October 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

   Internals needed to interoperate with the exception handling.  These are
   copied from the libsupc++ sources, but contain no functionality, only
   definitions.
*/

#ifndef __jml__arch__exception_internals_h__
#define __jml__arch__exception_internals_h__

#include <typeinfo>
#include <exception>
#include <cstddef>
#include <unwind.h>

namespace __cxxabiv1 {

// A C++ exception object consists of a header, which is a wrapper around
// an unwind object header with additional C++ specific information,
// followed by the exception object itself.

struct __cxa_exception
{ 
  // Manage the exception object itself.
  std::type_info *exceptionType;
  void (*exceptionDestructor)(void *); 

  // The C++ standard has entertaining rules wrt calling set_terminate
  // and set_unexpected in the middle of the exception cleanup process.
  std::unexpected_handler unexpectedHandler;
  std::terminate_handler terminateHandler;

  // The caught exception stack threads through here.
  __cxa_exception *nextException;

  // How many nested handlers have caught this exception.  A negated
  // value is a signal that this object has been rethrown.
  int handlerCount;

  // Cache parsed handler data from the personality routine Phase 1
  // for Phase 2 and __cxa_call_unexpected.
  int handlerSwitchValue;
  const unsigned char *actionRecord;
  const unsigned char *languageSpecificData;
  _Unwind_Ptr catchTemp;
  void *adjustedPtr;

  // The generic exception header.  Must be last.
  _Unwind_Exception unwindHeader;
};

struct __cxa_eh_globals
{
  __cxa_exception *caughtExceptions;
  unsigned int uncaughtExceptions;
};

extern "C" __cxa_eh_globals *__cxa_get_globals () throw();

} // __cxxabiv1


#endif /* __jml__arch__exception_internals_h__ */
