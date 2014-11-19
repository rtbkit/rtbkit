/* exception_hook.cc
   Jeremy Barnes, 7 February 2005
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
   
   Object file to install the exception tracer.
*/

#ifndef __arch__exception_hook_h__
#define __arch__exception_hook_h__

namespace ML {

/** Hook for the function to call when we throw an exception.  The first
    argument is a pointer to the object thrown; the second is the type
    info block for the thrown object.

    Starts off at null which means that no hook is installed.
*/
extern bool (*exception_tracer) (void *, const std::type_info *);

} // namespace ML

#endif /* __arch__exception_hook_h__ */
