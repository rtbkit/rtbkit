/* sgi_functional.h                                                -*- C++ -*-
   Jeremy Barnes, 2 February 2005
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
   
   Move SGI's <functional> algorithms into the std:: namespace.
*/

#ifndef __utils__sgi_functional_h__
#define __utils__sgi_functional_h__

#include <ext/functional>

namespace std {
    
using namespace __gnu_cxx;

};

#endif /* __utils__sgi_functional_h__ */
