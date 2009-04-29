/* threads.h                                                       -*- C++ -*-
   Jeremy Barnes, 30 January 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.

   This file is part of "Jeremy's Machine Learning Library", copyright (c)
   1999-2005 Jeremy Barnes.
   
   This program is available under the GNU General Public License, the terms
   of which are given by the file "license.txt" in the top level directory of
   the source code distribution.  If this file is missing, you have no right
   to use the program; please contact the author.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   ---

   Catch-all include for architecture dependent threading constructions.
*/

#ifndef __arch__threads_h__
#define __arch__threads_h__

#include <ace/Token.h>
#include <ace/Synch.h>


typedef ACE_Token Lock;
typedef ACE_Guard<Lock> Guard;
typedef ACE_Read_Guard<Lock> Read_Guard;
typedef ACE_Write_Guard<Lock> Write_Guard;


#endif /* __arch__threads_h__ */
