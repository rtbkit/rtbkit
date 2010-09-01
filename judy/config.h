/* config.h                                                        -*- C++ -*-
   Jeremy Barnes, 21 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$
   $Id$

   Configuration files for the judy array.
*/

#if defined(__amd64) // 64 bit machine; we need to compact them

#define JUDYL 1
#define JU_64BIT 1
#define JUDYERROR_NOTEST 1

#else

#define JUDYL 1
#define JU_32BIT 1
#define JUDYERROR_NOTEST 1

#endif
