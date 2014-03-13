/* py_boosting.cc
   Jeremy Barnes, 11 April 2005
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

   Overall file for boosting python wrappers.
*/

#include <boost/python.hpp>
#include "jml/boosting/training_data.h"
#include "jml/boosting/training_index.h"

using namespace boost::python;
using namespace ML;

namespace ML {
namespace Python {

void export_classifier();
void export_training_data();
void export_feature_set();

BOOST_PYTHON_MODULE(boosting)
{
    export_classifier();
    export_training_data();
    export_feature_set();
}

} // namespace Python
} // namespace ML
