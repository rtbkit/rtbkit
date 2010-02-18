/* environment.cc
   Jeremy Barnes, 14 March 2005
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
   
   Functions to deal with the environment.
*/

#include "jml/utils/environment.h"
#include <iostream>

using namespace std;


extern char ** environ;

namespace ML {

Environment::Environment()
{
    char ** e = environ;

    //cerr << "e = " << e << endl;

    while (e) {
        const char * p = *e++;
        if (!p) break;
        //cerr << "p = " << (void *)p << endl;
        //cerr << "p = " << p << endl;
        string s = p;
        //cerr << "s = " << s << endl;
        string::size_type equal_pos = s.find('=');
        if (equal_pos == string::npos) operator [] (s) = "";
        else {
            string key(s, 0, equal_pos);
            string val(s, equal_pos + 1);
            operator [] (key) = val;
        }
    }
}

const Environment &
Environment::instance()
{
    static const Environment result;
    return result;
}

} // namespace ML
