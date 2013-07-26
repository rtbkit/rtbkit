/* cpp11.h                                                -*- C++ -*-
   Janusz Lewandowski, 25 July 2013
   Copyright (c) 2013 Daftcode.

   C++11 features unimplemented in used gcc version.
*/

#pragma once

# if (defined __GNUC__) && (__GNUC__ < 5) && (__GNUC_MINOR__ < 7)
#define override
#endif
